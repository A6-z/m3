"""
M3 MCP Server - MIMIC-IV + MCP + Models
Provides MCP tools for querying MIMIC-IV data via DuckDB (local) or BigQuery.
"""

import os
from pathlib import Path

import duckdb
import sqlparse
from fastmcp import FastMCP

from m3.auth import init_oauth2, require_oauth2
from m3.config import get_default_database_path

# Create FastMCP server instance
mcp = FastMCP("m3")

# Global variables for backend configuration
_backend = None
_db_path = None
_bq_client = None
_project_id = None


def _validate_limit(limit: int) -> bool:
    """Validate limit parameter to prevent resource exhaustion."""
    return isinstance(limit, int) and 0 < limit <= 1000


def _is_safe_query(sql_query: str, internal_tool: bool = False) -> tuple[bool, str]:
    """Secure SQL validation - blocks injection attacks, allows legitimate queries."""
    try:
        if not sql_query or not sql_query.strip():
            return False, "Empty query"

        # Parse SQL to validate structure
        parsed = sqlparse.parse(sql_query.strip())
        if not parsed:
            return False, "Invalid SQL syntax"

        # Block multiple statements (main injection vector)
        if len(parsed) > 1:
            return False, "Multiple statements not allowed"

        statement = parsed[0]
        statement_type = statement.get_type()

        # Allow SELECT and PRAGMA (PRAGMA is needed for schema exploration)
        if statement_type not in (
            "SELECT",
            "UNKNOWN",
        ):  # PRAGMA shows as UNKNOWN in sqlparse
            return False, "Only SELECT and PRAGMA queries allowed"

        # Check if it's a PRAGMA statement (these are safe for schema exploration)
        sql_upper = sql_query.strip().upper()
        if sql_upper.startswith("PRAGMA"):
            return True, "Safe PRAGMA statement"

        # For SELECT statements, block dangerous injection patterns
        if statement_type == "SELECT":
            # Block dangerous write operations within SELECT
            dangerous_keywords = {
                "INSERT",
                "UPDATE",
                "DELETE",
                "DROP",
                "CREATE",
                "ALTER",
                "TRUNCATE",
                "REPLACE",
                "MERGE",
                "EXEC",
                "EXECUTE",
            }

            for keyword in dangerous_keywords:
                if f" {keyword} " in f" {sql_upper} ":
                    return False, f"Write operation not allowed: {keyword}"

            # Block common injection patterns that are rarely used in legitimate analytics
            injection_patterns = [
                # Classic SQL injection patterns
                ("1=1", "Classic injection pattern"),
                ("OR 1=1", "Boolean injection pattern"),
                ("AND 1=1", "Boolean injection pattern"),
                ("OR '1'='1'", "String injection pattern"),
                ("AND '1'='1'", "String injection pattern"),
                ("WAITFOR", "Time-based injection"),
                ("SLEEP(", "Time-based injection"),
                ("BENCHMARK(", "Time-based injection"),
                ("LOAD_FILE(", "File access injection"),
                ("INTO OUTFILE", "File write injection"),
                ("INTO DUMPFILE", "File write injection"),
            ]

            for pattern, description in injection_patterns:
                if pattern in sql_upper:
                    return False, f"Injection pattern detected: {description}"

        return True, "Safe"

    except Exception as e:
        return False, f"Validation error: {e}"


def _init_backend():
    """Initialize the backend based on environment variables."""
    global _backend, _db_path, _bq_client, _project_id

    # Initialize OAuth2 authentication
    init_oauth2()

    _backend = os.getenv("M3_BACKEND", "duckdb")

    if _backend == "duckdb":
        _db_path = os.getenv("M3_DB_PATH")
        if not _db_path:
            path = get_default_database_path("mimic-iv-demo")
            _db_path = str(path) if path else None
        if not _db_path or not Path(_db_path).exists():
            raise FileNotFoundError(f"DuckDB database not found: {_db_path}")

    elif _backend == "bigquery":
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "BigQuery dependencies not found. Install with: pip install google-cloud-bigquery"
            )

        # User's GCP project ID for authentication and billing
        # MIMIC-IV data resides in the public 'physionet-data' project
        _project_id = os.getenv("M3_PROJECT_ID", "physionet-data")
        try:
            _bq_client = bigquery.Client(project=_project_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BigQuery client: {e}")

    else:
        raise ValueError(f"Unsupported backend: {_backend}")


# Initialize backend when module is imported
_init_backend()


def _get_backend_info() -> str:
    """Get current backend information for display in responses."""
    if _backend == "duckdb":
        return f"🔧 **Current Backend:** DuckDB (local database)\n📁 **Database Path:** {_db_path}\n"
    else:
        return f"🔧 **Current Backend:** BigQuery (cloud database)\n☁️ **Project ID:** {_project_id}\n"


def _validate_note_type(note_type: str) -> bool:
    """Validate note_type parameter for clinical notes tools."""
    return note_type in {"all", "discharge", "radiology"}


def _escape_sql_literal(value: str) -> str:
    """Escape single quotes for safe SQL literal usage."""
    return value.replace("'", "''")


def _split_duckdb_table_name(table_name: str) -> tuple[str | None, str]:
    """Split DuckDB table name into optional schema + table parts."""
    clean = table_name.strip().strip("`").strip('"')
    if "." not in clean:
        return None, clean

    parts = [part.strip().strip('"') for part in clean.split(".") if part.strip()]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, clean


def _quote_duckdb_identifier(schema: str | None, table: str) -> str:
    """Build a safely quoted DuckDB table identifier."""
    if schema:
        return f'"{schema}"."{table}"'
    return f'"{table}"'


def _discover_notes_tables(note_type: str = "all") -> list[dict[str, str]]:
    """Discover clinical notes tables for active backend.

    Returns:
        List of dicts with keys: display_name, query_name, dataset, table_name
    """
    tables: list[dict[str, str]] = []

    if _backend == "duckdb":
        conn = duckdb.connect(_db_path)
        try:
            df = conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE (
                    LOWER(table_name) LIKE '%note%'
                    OR LOWER(table_name) LIKE '%discharge%'
                    OR LOWER(table_name) LIKE '%radiology%'
                  )
                ORDER BY table_schema, table_name
                """
            ).df()
        finally:
            conn.close()

        table_rows = (
            list(zip(df["table_schema"].tolist(), df["table_name"].tolist()))
            if not df.empty
            else []
        )
        for table_schema, table_name in table_rows:
            lower_name = table_name.lower()
            if note_type == "discharge" and "discharge" not in lower_name:
                continue
            if note_type == "radiology" and "radiology" not in lower_name:
                continue

            if table_schema and table_schema != "main":
                query_name = f'"{table_schema}"."{table_name}"'
                display_name = f"{table_schema}.{table_name}"
            else:
                query_name = f'"{table_name}"'
                display_name = table_name

            tables.append(
                {
                    "display_name": display_name,
                    "query_name": query_name,
                    "dataset": table_schema or "main",
                    "table_name": table_name,
                }
            )

        return tables

    # BigQuery: Dynamically discover all datasets, then search for notes tables
    try:
        datasets_df = _bq_client.query(
            """
            SELECT DISTINCT table_schema
            FROM `region-us.INFORMATION_SCHEMA.TABLE_STORAGE`
            ORDER BY table_schema
            """
        ).to_dataframe()
        bq_datasets = datasets_df["table_schema"].tolist() if not datasets_df.empty else []
    except Exception:
        # Fallback to known note datasets
        bq_datasets = [
            "physionet-data.mimiciv_note",
            "physionet-data.mimiciv_3_1_note",
        ]

    for dataset in bq_datasets:
        try:
            df = _bq_client.query(
                f"""
                SELECT table_name
                FROM `{dataset}.INFORMATION_SCHEMA.TABLES`
                ORDER BY table_name
                """
            ).to_dataframe()
        except Exception:
            continue

        if df.empty:
            continue

        for table_name in df["table_name"].tolist():
            lower_name = table_name.lower()
            if note_type == "discharge" and "discharge" not in lower_name:
                continue
            if note_type == "radiology" and "radiology" not in lower_name:
                continue
            if note_type == "all" and (
                "note" not in lower_name
                and "discharge" not in lower_name
                and "radiology" not in lower_name
            ):
                continue

            tables.append(
                {
                    "display_name": f"`{dataset}.{table_name}`",
                    "query_name": f"`{dataset}.{table_name}`",
                    "dataset": dataset,
                    "table_name": table_name,
                }
            )

    return tables


def _get_table_columns(table: dict[str, str]) -> list[str]:
    """Get ordered column names for a selected table."""
    if _backend == "duckdb":
        table_name = table["table_name"]
        table_schema = table.get("dataset", "main")
        conn = duckdb.connect(_db_path)
        try:
            if table_schema and table_schema != "main":
                pragma_target = f'"{table_schema}"."{table_name}"'
            else:
                pragma_target = f'"{table_name}"'
            df = conn.execute(f"PRAGMA table_info({pragma_target})").df()
        finally:
            conn.close()
        if df.empty:
            return []
        return df["name"].tolist()

    dataset = table["dataset"]
    table_name = table["table_name"]
    df = _bq_client.query(
        f"""
        SELECT column_name
        FROM `{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{_escape_sql_literal(table_name)}'
        ORDER BY ordinal_position
        """
    ).to_dataframe()
    if df.empty:
        return []
    return df["column_name"].tolist()


def _find_first_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find first matching column name from candidates (case-insensitive)."""
    mapping = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def _select_notes_table(
    discovered_tables: list[dict[str, str]], requested_table: str | None
) -> tuple[dict[str, str] | None, str | None]:
    """Select a notes table by optional user-provided table name."""
    if not discovered_tables:
        return None, "No clinical notes tables found"

    if not requested_table:
        return discovered_tables[0], None

    requested = requested_table.strip().strip("`").lower()
    for table in discovered_tables:
        display_clean = table["display_name"].strip("`").lower()
        query_clean = table["query_name"].strip("`").replace('"', "").lower()
        if (
            requested == display_clean
            or requested == table["table_name"].lower()
            or requested == query_clean
        ):
            return table, None

    options = "\n".join(f"- {t['display_name']}" for t in discovered_tables)
    return None, f"Table '{requested_table}' not found.\n\n📋 **Available notes tables:**\n{options}"


# ==========================================
# INTERNAL QUERY EXECUTION FUNCTIONS
# ==========================================
# These functions perform the actual database operations
# and are called by the MCP tools. This prevents MCP tools
# from calling other MCP tools, which violates the MCP protocol.


def _execute_duckdb_query(sql_query: str) -> str:
    """Execute DuckDB query - internal function."""
    try:
        conn = duckdb.connect(_db_path)
        try:
            df = conn.execute(sql_query).df()
            if df.empty:
                return "No results found"
            if len(df) > 50:
                out = (
                    df.head(50).to_string(index=False)
                    + f"\n... ({len(df)} total rows, showing first 50)"
                )
            else:
                out = df.to_string(index=False)
            return out
        finally:
            conn.close()
    except Exception as e:
        # Re-raise the exception so the calling function can handle it with enhanced guidance
        raise e


def _execute_bigquery_query(sql_query: str) -> str:
    """Execute BigQuery query - internal function."""
    try:
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()
        query_job = _bq_client.query(sql_query, job_config=job_config)
        df = query_job.to_dataframe()

        if df.empty:
            return "No results found"

        # Limit output size
        if len(df) > 50:
            result = df.head(50).to_string(index=False)
            result += f"\n... ({len(df)} total rows, showing first 50)"
        else:
            result = df.to_string(index=False)

        return result

    except Exception as e:
        # Re-raise the exception so the calling function can handle it with enhanced guidance
        raise e


def _execute_query_internal(sql_query: str) -> str:
    """Internal query execution function that handles backend routing."""
    # Security check
    is_safe, message = _is_safe_query(sql_query)
    if not is_safe:
        if "describe" in sql_query.lower() or "show" in sql_query.lower():
            return f"""❌ **Security Error:** {message}

        🔍 **For table structure:** Use `get_table_info('table_name')` instead of DESCRIBE
        📋 **Why this is better:** Shows columns, types, AND sample data to understand the actual data

        💡 **Recommended workflow:**
        1. `get_database_schema()` ← See available tables
        2. `get_table_info('table_name')` ← Explore structure
        3. `execute_mimic_query('SELECT ...')` ← Run your analysis"""

        return f"❌ **Security Error:** {message}\n\n💡 **Tip:** Only SELECT statements are allowed for data analysis."

    try:
        if _backend == "duckdb":
            return _execute_duckdb_query(sql_query)
        else:  # bigquery
            return _execute_bigquery_query(sql_query)
    except Exception as e:
        error_msg = str(e).lower()

        # Provide specific, actionable error guidance
        suggestions = []

        if "no such table" in error_msg or "table not found" in error_msg:
            suggestions.append(
                "🔍 **Table name issue:** Use `get_database_schema()` to see exact table names"
            )
            suggestions.append(
                f"📋 **Backend-specific naming:** {_backend} has specific table naming conventions"
            )
            suggestions.append(
                "💡 **Quick fix:** Check if the table name matches exactly (case-sensitive)"
            )

        if "no such column" in error_msg or "column not found" in error_msg:
            suggestions.append(
                "🔍 **Column name issue:** Use `get_table_info('table_name')` to see available columns"
            )
            suggestions.append(
                "📝 **Common issue:** Column might be named differently (e.g., 'anchor_age' not 'age')"
            )
            suggestions.append(
                "👀 **Check sample data:** `get_table_info()` shows actual column names and sample values"
            )

        if "syntax error" in error_msg:
            suggestions.append(
                "📝 **SQL syntax issue:** Check quotes, commas, and parentheses"
            )
            suggestions.append(
                f"🎯 **Backend syntax:** Verify your SQL works with {_backend}"
            )
            suggestions.append(
                "💭 **Try simpler:** Start with `SELECT * FROM table_name LIMIT 5`"
            )

        if "describe" in error_msg.lower() or "show" in error_msg.lower():
            suggestions.append(
                "🔍 **Schema exploration:** Use `get_table_info('table_name')` instead of DESCRIBE"
            )
            suggestions.append(
                "📋 **Better approach:** `get_table_info()` shows columns AND sample data"
            )

        if not suggestions:
            suggestions.append(
                "🔍 **Start exploration:** Use `get_database_schema()` to see available tables"
            )
            suggestions.append(
                "📋 **Check structure:** Use `get_table_info('table_name')` to understand the data"
            )

        suggestion_text = "\n".join(f"   {s}" for s in suggestions)

        return f"""❌ **Query Failed:** {e}

🛠️ **How to fix this:**
{suggestion_text}

🎯 **Quick Recovery Steps:**
1. `get_database_schema()` ← See what tables exist
2. `get_table_info('your_table')` ← Check exact column names
3. Retry your query with correct names

📚 **Current Backend:** {_backend} - table names and syntax are backend-specific"""


# ==========================================
# MCP TOOLS - PUBLIC API
# ==========================================
# These are the tools exposed via MCP protocol.
# They should NEVER call other MCP tools - only internal functions.


@mcp.tool()
@require_oauth2
def get_database_schema() -> str:
    """🔍 Discover what data is available in the MIMIC-IV database (and any custom tables).

    **When to use:** Start here when you need to understand what tables exist, or when someone asks about data that might be in multiple tables.

    **What this does:** 
    - Shows all available MIMIC-IV tables
    - Shows any custom/user-created tables in your project
    - Lists tables in database schema format

    **Key info:** This tool discovers ALL tables accessible to you, including custom ones like `patient_notes_json`.
    If you see a table here, you should be able to:
    1. Query it directly with `execute_mimic_query()` using its fully qualified name
    2. Explore it with `get_table_info()` if the name matches exactly
    3. Join it with other tables in your SQL queries

    **Next steps after using this:**
    - If you see relevant tables, use `get_table_info(table_name)` to explore their structure
    - For tables not found by `get_table_info()`, use `execute_mimic_query()` with fully qualified name
    - Common MIMIC-IV tables: `patients` (demographics), `admissions` (hospital stays), `icustays` (ICU data), `labevents` (lab results)

    Returns:
        List of all available tables in the database with current backend info
    """
    if _backend == "duckdb":
        query = """
        SELECT
            CASE
                WHEN table_schema = 'main' THEN table_name
                ELSE table_schema || '.' || table_name
            END AS table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_name
        """
        result = _execute_query_internal(query)
        return f"{_get_backend_info()}\n📋 **Available Tables:**\n{result}"

    elif _backend == "bigquery":
        # Dynamically discover all datasets accessible to the project
        try:
            datasets_df = _bq_client.query(
                """
                SELECT DISTINCT table_schema
                FROM `region-us.INFORMATION_SCHEMA.TABLE_STORAGE`
                ORDER BY table_schema
                """
            ).to_dataframe()
            dataset_list = (
                datasets_df["table_schema"].tolist()
                if not datasets_df.empty
                else []
            )
        except Exception:
            # Fallback to known MIMIC datasets if dynamic discovery fails
            dataset_list = [
                "physionet-data.mimiciv_3_1_hosp",
                "physionet-data.mimiciv_3_1_icu",
                "physionet-data.mimiciv_note",
                "physionet-data.mimiciv_3_1_note",
            ]

        all_tables: list[str] = []
        for dataset in dataset_list:
            try:
                df = _bq_client.query(
                    f"""
                    SELECT table_name
                    FROM `{dataset}.INFORMATION_SCHEMA.TABLES`
                    ORDER BY table_name
                    """
                ).to_dataframe()
            except Exception:
                continue

            if not df.empty:
                all_tables.extend([f"`{dataset}.{name}`" for name in df["table_name"]])

        if not all_tables:
            return f"{_get_backend_info()}\n📋 **Available Tables (query-ready names):**\nNo tables found"

        result = "\n".join(all_tables)
        return f"{_get_backend_info()}\n📋 **Available Tables (query-ready names):**\n{result}\n\n💡 **Copy-paste ready:** These table names can be used directly in your SQL queries!\n💡 **Custom tables:** Any tables you've created in your project (like patient_notes_json) will appear here."


@mcp.tool()
@require_oauth2
def get_table_info(table_name: str, show_sample: bool = True) -> str:
    """📋 Explore a specific table's structure and see sample data.

    **When to use:** After you know which table you need (from `get_database_schema()`), use this to understand the columns and data format.

    **What this does:**
    - Shows column names, types, and constraints
    - Displays sample rows so you understand the actual data format
    - Helps you write accurate SQL queries

    **Works with:**
    - MIMIC-IV tables (e.g., `patients`, `admissions`)
    - Custom/user-created tables (e.g., `patient_notes_json`)
    - Fully qualified BigQuery names (e.g., `my-project.my_dataset.my_table`)

    **Pro tip:** Always look at sample data! It shows you the actual values, date formats, and data patterns.

    **For custom tables in another GCP project:**
    If your table is in a different GCP project and `get_database_schema()` shows it but this function fails:
    1. Use the fully qualified name: `your-project.your_dataset.table_name`
    2. Or set `M3_PROJECT_ID=your-project` environment variable and restart

    Args:
        table_name: Table name - can be simple (patients), fully qualified BigQuery name (project.dataset.table), or from get_database_schema() output
        show_sample: Whether to include sample rows (default: True, recommended)

    Returns:
        Complete table structure with sample data to help you write queries
    """
    backend_info = _get_backend_info()

    if _backend == "duckdb":
        schema_name, plain_table_name = _split_duckdb_table_name(table_name)
        qualified_name = _quote_duckdb_identifier(schema_name, plain_table_name)

        # Get column information
        pragma_query = f"PRAGMA table_info({qualified_name})"
        try:
            result = _execute_duckdb_query(pragma_query)
            if "error" in result.lower():
                return f"{backend_info}❌ Table '{table_name}' not found. Use get_database_schema() to see available tables."

            shown_name = (
                f"{schema_name}.{plain_table_name}" if schema_name else plain_table_name
            )
            info_result = f"{backend_info}📋 **Table:** {shown_name}\n\n**Column Information:**\n{result}"

            if show_sample:
                sample_query = f"SELECT * FROM {qualified_name} LIMIT 3"
                sample_result = _execute_duckdb_query(sample_query)
                info_result += (
                    f"\n\n📊 **Sample Data (first 3 rows):**\n{sample_result}"
                )

            return info_result
        except Exception as e:
            return f"{backend_info}❌ Error examining table '{table_name}': {e}\n\n💡 Use get_database_schema() to see available tables."

    else:  # bigquery
        # Try three approaches: (1) fully qualified name, (2) simple name with dynamic dataset search, (3) fallback
        simple_table_name = table_name.strip("`").split(".")[-1] if "." in table_name else table_name.strip("`")
        
        # Attempt 1: If user provided fully qualified name, use it directly
        if "." in table_name.strip("`"):
            clean_name = table_name.strip("`")
            full_table_name = f"`{clean_name}`"
            parts = clean_name.split(".")
            
            if len(parts) == 3:
                project, dataset_part, table_part = parts
                dataset = f"`{project}.{dataset_part}`"
                simple_table_name = table_part
                
                try:
                    info_query = f"""
                    SELECT column_name, data_type, is_nullable
                    FROM {dataset}.INFORMATION_SCHEMA.COLUMNS
                    WHERE table_name = '{simple_table_name}'
                    ORDER BY ordinal_position
                    """
                    
                    info_result = _execute_bigquery_query(info_query)
                    if "No results found" not in info_result:
                        result = f"{backend_info}📋 **Table:** {full_table_name}\n\n**Column Information:**\n{info_result}"
                        
                        if show_sample:
                            sample_query = f"SELECT * FROM {full_table_name} LIMIT 3"
                            sample_result = _execute_bigquery_query(sample_query)
                            result += f"\n\n📊 **Sample Data (first 3 rows):**\n{sample_result}"
                        
                        return result
                except Exception as e:
                    pass  # Fall through to dynamic search
        
        # Attempt 2: Dynamic dataset search for simple table names
        # Query all accessible datasets and search for the table
        try:
            search_query = f"""
            SELECT DISTINCT table_schema
            FROM `region-us.INFORMATION_SCHEMA.TABLE_STORAGE`
            ORDER BY table_schema
            """
            datasets_df = _bq_client.query(search_query).to_dataframe()
            dataset_list = datasets_df["table_schema"].tolist() if not datasets_df.empty else []
        except Exception:
            # Fallback to known MIMIC and common datasets
            dataset_list = [
                "physionet-data.mimiciv_3_1_hosp",
                "physionet-data.mimiciv_3_1_icu",
                "physionet-data.mimiciv_note",
                "physionet-data.mimiciv_3_1_note",
            ]
        
        # Search for the table in all datasets
        for dataset in dataset_list:
            try:
                full_table_name = f"`{dataset}.{simple_table_name}`"
                info_query = f"""
                SELECT column_name, data_type, is_nullable
                FROM `{dataset}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{simple_table_name}'
                ORDER BY ordinal_position
                """
                
                info_result = _execute_bigquery_query(info_query)
                if "No results found" not in info_result:
                    result = f"{backend_info}📋 **Table:** {full_table_name}\n\n**Column Information:**\n{info_result}"
                    
                    if show_sample:
                        sample_query = f"SELECT * FROM {full_table_name} LIMIT 3"
                        sample_result = _execute_bigquery_query(sample_query)
                        result += f"\n\n📊 **Sample Data (first 3 rows):**\n{sample_result}"
                    
                    return result
            except Exception:
                continue
        
        return f"{backend_info}❌ Table '{table_name}' not found. 

**Troubleshooting:**
1. Ensure you provided the correct table name (case matters in BigQuery)
2. Use `get_database_schema()` to see the exact fully qualified table name
3. If your custom table is in a different GCP project, provide the fully qualified name: `project.dataset.table_name`
4. For custom tables in your own project, set the environment variable: `M3_PROJECT_ID=your-gcp-project` and restart

**Example for custom table:**
   `get_table_info('your-project.your_dataset.patient_notes_json')`"


@mcp.tool()
@require_oauth2
def execute_mimic_query(sql_query: str) -> str:
    """🚀 Execute SQL queries to analyze MIMIC-IV data (or any custom tables).

    **💡 Pro tip:** For best results, explore the database structure first!

    **Recommended workflow (especially for smaller models):**
    1. **See available tables:** Use `get_database_schema()` to list all tables (including custom ones)
    2. **Examine table structure:** Use `get_table_info('table_name')` to see columns and sample data
    3. **Write your SQL query:** Use exact table/column names from exploration

    **Why exploration helps:**
    - Table names vary between backends (DuckDB vs BigQuery)
    - Column names may be unexpected (e.g., age might be 'anchor_age')
    - Sample data shows actual formats and constraints

    **Works with:**
    - MIMIC-IV tables ✅
    - Custom/user-created tables ✅
    - Joined queries across multiple tables ✅
    - ANY table visible in `get_database_schema()` ✅

    **If `get_table_info()` can't find your custom table:**
    You can still query it directly with `execute_mimic_query()` using its fully qualified name:
    ```sql
    SELECT * FROM `your-project.your_dataset.patient_notes_json` LIMIT 10
    ```

    Args:
        sql_query: Your SQL SELECT query (must be SELECT only for security)

    Returns:
        Query results or helpful error messages with next steps
    """
    return _execute_query_internal(sql_query)


@mcp.tool()
@require_oauth2
def get_icu_stays(patient_id: int | None = None, limit: int = 10) -> str:
    """🏥 Get ICU stay information and length of stay data.

    **⚠️ Note:** This is a convenience function that assumes standard MIMIC-IV table structure.
    **For reliable queries:** Use `get_database_schema()` → `get_table_info()` → `execute_mimic_query()` workflow.

    **What you'll get:** Patient IDs, admission times, length of stay, and ICU details.

    Args:
        patient_id: Specific patient ID to query (optional)
        limit: Maximum number of records to return (default: 10)

    Returns:
        ICU stay data as formatted text or guidance if table not found
    """
    # Security validation
    if not _validate_limit(limit):
        return "Error: Invalid limit. Must be a positive integer between 1 and 10000."

    # Try common ICU table names based on backend
    if _backend == "duckdb":
        icustays_table = "icu_icustays"
    else:  # bigquery
        icustays_table = "`physionet-data.mimiciv_3_1_icu.icustays`"

    if patient_id:
        query = f"SELECT * FROM {icustays_table} WHERE subject_id = {patient_id}"
    else:
        query = f"SELECT * FROM {icustays_table} LIMIT {limit}"

    # Execute with error handling that suggests proper workflow
    result = _execute_query_internal(query)
    if "error" in result.lower() or "not found" in result.lower():
        return f"""❌ **Convenience function failed:** {result}

💡 **For reliable results, use the proper workflow:**
1. `get_database_schema()` ← See actual table names
2. `get_table_info('table_name')` ← Understand structure
3. `execute_mimic_query('your_sql')` ← Use exact names

This ensures compatibility across different MIMIC-IV setups."""

    return result


@mcp.tool()
@require_oauth2
def get_lab_results(
    patient_id: int | None = None, lab_item: str | None = None, limit: int = 20
) -> str:
    """🧪 Get laboratory test results quickly.

    **⚠️ Note:** This is a convenience function that assumes standard MIMIC-IV table structure.
    **For reliable queries:** Use `get_database_schema()` → `get_table_info()` → `execute_mimic_query()` workflow.

    **What you'll get:** Lab values, timestamps, patient IDs, and test details.

    Args:
        patient_id: Specific patient ID to query (optional)
        lab_item: Lab item to search for in the value field (optional)
        limit: Maximum number of records to return (default: 20)

    Returns:
        Lab results as formatted text or guidance if table not found
    """
    # Security validation
    if not _validate_limit(limit):
        return "Error: Invalid limit. Must be a positive integer between 1 and 10000."

    # Try common lab table names based on backend
    if _backend == "duckdb":
        labevents_table = "hosp_labevents"
    else:  # bigquery
        labevents_table = "`physionet-data.mimiciv_3_1_hosp.labevents`"

    # Build query conditions
    conditions = []
    if patient_id:
        conditions.append(f"subject_id = {patient_id}")
    if lab_item:
        # Escape single quotes for safety in LIKE clause
        escaped_lab_item = lab_item.replace("'", "''")
        conditions.append(f"value LIKE '%{escaped_lab_item}%'")

    base_query = f"SELECT * FROM {labevents_table}"
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)
    base_query += f" LIMIT {limit}"

    # Execute with error handling that suggests proper workflow
    result = _execute_query_internal(base_query)
    if "error" in result.lower() or "not found" in result.lower():
        return f"""❌ **Convenience function failed:** {result}

💡 **For reliable results, use the proper workflow:**
1. `get_database_schema()` ← See actual table names
2. `get_table_info('table_name')` ← Understand structure
3. `execute_mimic_query('your_sql')` ← Use exact names

This ensures compatibility across different MIMIC-IV setups."""

    return result


@mcp.tool()
@require_oauth2
def get_race_distribution(limit: int = 10) -> str:
    """📊 Get race distribution from hospital admissions.

    **⚠️ Note:** This is a convenience function that assumes standard MIMIC-IV table structure.
    **For reliable queries:** Use `get_database_schema()` → `get_table_info()` → `execute_mimic_query()` workflow.

    **What you'll get:** Count of patients by race category, ordered by frequency.

    Args:
        limit: Maximum number of race categories to return (default: 10)

    Returns:
        Race distribution as formatted text or guidance if table not found
    """
    # Security validation
    if not _validate_limit(limit):
        return "Error: Invalid limit. Must be a positive integer between 1 and 10000."

    # Try common admissions table names based on backend
    if _backend == "duckdb":
        admissions_table = "hosp_admissions"
    else:  # bigquery
        admissions_table = "`physionet-data.mimiciv_3_1_hosp.admissions`"

    query = f"SELECT race, COUNT(*) as count FROM {admissions_table} GROUP BY race ORDER BY count DESC LIMIT {limit}"

    # Execute with error handling that suggests proper workflow
    result = _execute_query_internal(query)
    if "error" in result.lower() or "not found" in result.lower():
        return f"""❌ **Convenience function failed:** {result}

💡 **For reliable results, use the proper workflow:**
1. `get_database_schema()` ← See actual table names
2. `get_table_info('table_name')` ← Understand structure
3. `execute_mimic_query('your_sql')` ← Use exact names

This ensures compatibility across different MIMIC-IV setups."""

    return result


@mcp.tool()
@require_oauth2
def get_clinical_notes_table(
    table_name: str | None = None,
    note_type: str = "all",
    limit: int = 5,
    preview_chars: int = 300,
) -> str:
    """📝 Explore clinical notes rows with linked IDs for cross-tool workflows.

    This tool focuses on notes-table access and returns row-level note previews
    plus relation IDs (subject_id/hadm_id/stay_id when available) so results can
    be joined with other MIMIC tools.

    Args:
        table_name: Optional specific notes table. If omitted, first discovered table is used.
        note_type: Filter tables by type ('all', 'discharge', or 'radiology').
        limit: Number of rows to preview.
        preview_chars: Max characters shown from note text per row.

    Returns:
        Available notes tables, selected table, schema summary, and sample rows.
    """
    if not _validate_limit(limit):
        return "Error: Invalid limit. Must be a positive integer between 1 and 1000."

    if not isinstance(preview_chars, int) or preview_chars <= 0 or preview_chars > 5000:
        return "Error: Invalid preview_chars. Must be an integer between 1 and 5000."

    if not _validate_note_type(note_type):
        return "Error: Invalid note_type. Use one of: 'all', 'discharge', 'radiology'."

    backend_info = _get_backend_info()

    try:
        discovered_tables = _discover_notes_tables(note_type=note_type)
    except Exception as e:
        return f"{backend_info}❌ Failed to discover notes tables: {e}"

    selected_table, selection_error = _select_notes_table(discovered_tables, table_name)
    if selection_error:
        return f"{backend_info}❌ {selection_error}"
    if not selected_table:
        return f"{backend_info}❌ No clinical notes tables available."

    available_tables_text = "\n".join(
        f"- {t['display_name']}" for t in discovered_tables
    )

    try:
        columns = _get_table_columns(selected_table)
    except Exception as e:
        return f"{backend_info}❌ Failed to read table schema for {selected_table['display_name']}: {e}"

    if not columns:
        return f"{backend_info}❌ Could not find columns for {selected_table['display_name']}."

    id_candidates = [
        "note_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "chartdate",
        "storetime",
        "note_type",
        "category",
    ]
    text_candidates = ["text", "note_text", "note", "content"]

    selected_id_columns = [c for c in id_candidates if c in {x.lower() for x in columns}]
    resolved_id_columns = []
    for candidate in selected_id_columns:
        col_name = _find_first_column(columns, [candidate])
        if col_name:
            resolved_id_columns.append(col_name)

    text_column = _find_first_column(columns, text_candidates)
    if not text_column:
        return (
            f"{backend_info}❌ Could not detect a note text column in {selected_table['display_name']}.\n\n"
            f"Detected columns: {', '.join(columns)}"
        )

    projection_parts = resolved_id_columns.copy()
    if _backend == "duckdb":
        projection_parts.append(
            f"SUBSTR(CAST({text_column} AS VARCHAR), 1, {preview_chars}) AS note_preview"
        )
    else:
        projection_parts.append(
            f"SUBSTR(CAST({text_column} AS STRING), 1, {preview_chars}) AS note_preview"
        )

    query = (
        f"SELECT {', '.join(projection_parts)} "
        f"FROM {selected_table['query_name']} "
        f"LIMIT {limit}"
    )

    rows = _execute_query_internal(query)
    if "❌" in rows:
        return rows

    relation_keys = ", ".join(
        [c for c in ["subject_id", "hadm_id", "stay_id", "note_id"] if c in {x.lower() for x in columns}]
    )
    if not relation_keys:
        relation_keys = "No standard relation IDs detected"

    return (
        f"{backend_info}"
        f"📋 **Available Notes Tables:**\n{available_tables_text}\n\n"
        f"🧾 **Selected Table:** {selected_table['display_name']}\n"
        f"🔗 **Detected Relation Keys:** {relation_keys}\n"
        f"📝 **Detected Text Column:** {text_column}\n\n"
        f"📊 **Sample Note Rows (limit {limit}):**\n{rows}\n\n"
        "💡 Use detected relation keys (subject_id/hadm_id/stay_id) with other tools like `get_icu_stays`, `get_lab_results`, and `execute_mimic_query` to connect notes with structured data."
    )


@mcp.tool()
@require_oauth2
def get_clinical_note_row(
    note_id: str,
    table_name: str | None = None,
    note_type: str = "all",
    max_note_chars: int = 4000,
) -> str:
    """📄 Read a specific clinical note row with its linked identifiers.

    This tool retrieves one row from a notes table by note identifier and returns
    relation IDs plus note text to support downstream linking with other tools.

    Args:
        note_id: Note identifier value to search.
        table_name: Optional specific notes table.
        note_type: Filter notes tables by type ('all', 'discharge', 'radiology').
        max_note_chars: Maximum note text characters to return.

    Returns:
        A single note row containing identifiers and note content preview/full text.
    """
    if not note_id or not note_id.strip():
        return "Error: note_id is required."

    if not _validate_note_type(note_type):
        return "Error: Invalid note_type. Use one of: 'all', 'discharge', 'radiology'."

    if (
        not isinstance(max_note_chars, int)
        or max_note_chars <= 0
        or max_note_chars > 50000
    ):
        return "Error: Invalid max_note_chars. Must be an integer between 1 and 50000."

    backend_info = _get_backend_info()

    try:
        discovered_tables = _discover_notes_tables(note_type=note_type)
    except Exception as e:
        return f"{backend_info}❌ Failed to discover notes tables: {e}"

    selected_table, selection_error = _select_notes_table(discovered_tables, table_name)
    if selection_error:
        return f"{backend_info}❌ {selection_error}"
    if not selected_table:
        return f"{backend_info}❌ No clinical notes tables available."

    try:
        columns = _get_table_columns(selected_table)
    except Exception as e:
        return f"{backend_info}❌ Failed to read table schema for {selected_table['display_name']}: {e}"

    if not columns:
        return f"{backend_info}❌ Could not find columns for {selected_table['display_name']}."

    note_id_column = _find_first_column(columns, ["note_id", "row_id", "id"])
    if not note_id_column:
        return (
            f"{backend_info}❌ Could not detect note ID column in {selected_table['display_name']}.\n\n"
            f"Detected columns: {', '.join(columns)}"
        )

    text_column = _find_first_column(columns, ["text", "note_text", "note", "content"])
    if not text_column:
        return (
            f"{backend_info}❌ Could not detect note text column in {selected_table['display_name']}.\n\n"
            f"Detected columns: {', '.join(columns)}"
        )

    projection = []
    for col in ["note_id", "subject_id", "hadm_id", "stay_id", "charttime", "chartdate", "note_type", "category"]:
        matched = _find_first_column(columns, [col])
        if matched:
            projection.append(matched)

    if _backend == "duckdb":
        projection.append(
            f"SUBSTR(CAST({text_column} AS VARCHAR), 1, {max_note_chars}) AS note_text"
        )
        where_clause = (
            f"CAST({note_id_column} AS VARCHAR) = '{_escape_sql_literal(note_id.strip())}'"
        )
    else:
        projection.append(
            f"SUBSTR(CAST({text_column} AS STRING), 1, {max_note_chars}) AS note_text"
        )
        where_clause = (
            f"CAST({note_id_column} AS STRING) = '{_escape_sql_literal(note_id.strip())}'"
        )

    query = (
        f"SELECT {', '.join(projection)} "
        f"FROM {selected_table['query_name']} "
        f"WHERE {where_clause} "
        "LIMIT 1"
    )

    row_result = _execute_query_internal(query)
    if "No results found" in row_result:
        return (
            f"{backend_info}❌ No note found for note_id='{note_id}' in {selected_table['display_name']}.\n\n"
            "💡 Use `get_clinical_notes_table()` first to inspect available IDs and tables."
        )
    if "❌" in row_result:
        return row_result

    return (
        f"{backend_info}"
        f"🧾 **Table:** {selected_table['display_name']}\n"
        f"🔎 **Lookup:** {note_id_column} = '{note_id.strip()}'\n\n"
        f"📄 **Note Row:**\n{row_result}\n\n"
        "💡 You can use `subject_id`, `hadm_id`, or `stay_id` from this row with other MIMIC tools to connect structured and unstructured data."
    )


def main():
    """Main entry point for MCP server.

    Runs FastMCP server in either STDIO mode (desktop clients) or HTTP mode
    (Kubernetes/web clients). Transport mode configured via environment variables.

    Environment Variables:
        MCP_TRANSPORT: "stdio" (default), "sse", or "http"
        MCP_HOST: Host binding for HTTP mode (default: "0.0.0.0")
        MCP_PORT: Port for HTTP mode (default: 3000)
        MCP_PATH: SSE endpoint path for HTTP mode (default: "/sse")

    Notes:
        HTTP/SSE mode uses streamable-http transport for containerized deployments
        where STDIO is unavailable. Binds to 0.0.0.0 for Kubernetes service mesh access.
    """
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "3000"))
    path = os.getenv("MCP_PATH", "/sse")
    mcp.run(transport="streamable-http", host=host, port=port, path=path)



if __name__ == "__main__":
    main()
