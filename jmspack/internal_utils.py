"""Submodule internal_utils.py includes the following functions: <br>
- postgresql_data_extraction():
- postgresql_table_names_list():
"""
import os

import pandas as pd
import psycopg2


def postgresql_data_extraction(
    table_name: str = "suggested_energy_intake",
    database_name: str = "tracker",
    user: str = "tracker",
):
    r"""
    Load data from a specified postgresql database.

    Parameters
    ----------
    table_name: str
        The name of the table to extract from the postgresql database.
    database_name: str
        The name of the postgresql database.
    user: str
        The name of the user.

    Returns
    -------
    pd.DataFrame

    Examples
    ---------
    >>> from jmspack.internal_utils import postgresql_data_extraction
    >>> df = postgresql_data_extraction()
    """
    df = pd.DataFrame()
    try:
        conn = psycopg2.connect(
            host=os.getenv("postgresql_host"),
            database=database_name,
            user=user,
            password=os.getenv("postgresql_password"),
        )
        df = pd.read_sql_query(f"SELECT * from {table_name}", conn)
        _ = conn.close()

    except:
        print("I am unable to connect to the database")

    return df


def postgresql_table_names_list(
    database_name: str = "tracker",
    user="tracker",
):
    r"""
    Extract the table names from a specified postgresql database.

    Parameters
    ----------
    database_name: str
        The name of the postgresql database.
    user: str
        The name of the user.

    Returns
    -------
    list

    Examples
    ---------
    >>> from jmspack.internal_utils import postgresql_table_names_list
    >>> table_names = postgresql_table_names_list()
    """
    table_list = False
    try:
        conn = psycopg2.connect(
            host=os.getenv("postgresql_host"),
            database=database_name,
            user=user,
            password=os.getenv("postgresql_password"),
        )
        cursor = conn.cursor()
        cursor.execute(
            "select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';"
        )
        table_list = cursor.fetchall()
        cursor.close()
        _ = conn.close()
    except psycopg2.Error as e:
        print(e)
    return table_list


def create_postgresql_table_based_on_df(
    df: pd.DataFrame,
    database_name: str,
    user: str,
    table_name: str,
):
    python_to_sql_dtypes_dict = {
        "object": "text",
        "float64": "float",
        "float32": "float",
        "int8": "int",
        "int32": "int",
        "int64": "int",
        "datetime64[ns]": "timestamp",
    }
    df_dtypes_dict = df.dtypes.to_dict()
    columns_dtypes_list = [
        f"{column} {python_to_sql_dtypes_dict[str(df_dtypes_dict[column])]}"
        for column in df.columns.tolist()
    ]
    columns_dtypes_str = ",\n".join(columns_dtypes_list)
    table_string = f"""CREATE TABLE {table_name} (
                {columns_dtypes_str}
    )"""

    try:
        conn = psycopg2.connect(
            host=os.getenv("postgresql_host"),
            database=database_name,
            user=user,
            password=os.getenv("postgresql_password"),
        )
        cursor = conn.cursor()
        _ = cursor.execute(table_string)
        _ = conn.commit()
        _ = conn.close()
    except psycopg2.Error as e:
        print(e)

    return table_string


def add_data_to_postgresql_table(
    df: pd.DataFrame,
    database_name: str,
    user: str,
    table_name: str,
):
    columns_string = ", ".join(df.columns.tolist())
    value_placeholder_string = ", ".join(["%s" for x in range(0, df.shape[1])])
    insert_string = f"""INSERT INTO {table_name} ({columns_string}) VALUES ({value_placeholder_string})"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("postgresql_host"),
            database=database_name,
            user=user,
            password=os.getenv("postgresql_password"),
        )
        cursor = conn.cursor()
        for oid in range(0, df.shape[0]):
            value_tuple = tuple(
                df.loc[oid, df.columns[col_number]]
                for col_number in range(0, df.shape[1])
            )
            _ = cursor.execute(insert_string, value_tuple)

        _ = conn.commit()
        _ = conn.close()

    except psycopg2.Error as e:
        print(e)

    return insert_string


def delete_postgresql_table(
    database_name: str,
    user: str,
    table_name: str,
):
    try:
        conn = psycopg2.connect(
            host=os.getenv("postgresql_host"),
            database=database_name,
            user=user,
            password=os.getenv("postgresql_password"),
        )
        cursor = conn.cursor()
        _ = cursor.execute("DROP TABLE iris_test;")
        _ = conn.commit()
        _ = conn.close()
    except psycopg2.Error as e:
        print(e)

    return f"{table_name} has been deleted from {database_name}"
