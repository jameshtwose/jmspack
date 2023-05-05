r"""Submodule internal_utils.py includes the following functions:

- **postgresql_data_extraction():** importing data from a table of a postgresql database.

- **postgresql_table_names_list():** extract the table names from a specified postgresql database.

- **create_postgresql_table_based_on_df():** create a new table in a specified postgresql database based on the columns of a pandas data frame.

- **add_data_to_postgresql_table():** add new data to an existing table in a specified postgresql database.

- **delete_postgresql_table():** delete a table from a postgresql database.

"""
import os

import pandas as pd
import psycopg2

# INFO: for all of postgresql based functions, you will need to specify your
# credentials in a ".env" locally. The two required parameters are:
# postgresql_host="BLA" (the host value for the postgresql database)
# postgresql_password="BLA2" (the password value for the postgresql database)


def postgresql_data_extraction(
    table_name: str = "suggested_energy_intake",
    database_name: str = "tracker",
    user: str = "tracker",
):
    """Load data from a specified postgresql database.

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
    df: pd.DataFrame
        pandas dataframe object containing the data from the specified table.

    Examples
    --------
    >>> from dotenv import load_dotenv, find_dotenv
    >>> from jmspack.internal_utils import postgresql_data_extraction
    >>> # Make sure you have a .env file somewhere with your postgresql credentials
    >>> # labelled as postgresql_host="BLA", and postgresql_password="BLA2"
    >>> load_dotenv(find_dotenv())
    >>> df = postgresql_data_extraction(table_name = 'iris_test',
    ...                            database_name = 'tracker',
    ...                            user='tracker')

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

    except psycopg2.errors.lookup("08006"):
        print("I am unable to connect to the database")

    return df


def postgresql_table_names_list(
    database_name: str = "tracker",
    user="tracker",
):
    """Extract the table names from a specified postgresql database.

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
    --------
    >>> from dotenv import load_dotenv, find_dotenv
    >>> from jmspack.internal_utils import postgresql_table_names_list
    >>> # Make sure you have a .env file somewhere with your postgresql credentials
    >>> # labelled as postgresql_host="BLA", and postgresql_password="BLA2"
    >>> load_dotenv(find_dotenv())
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

    """Create a new table in a specified postgresql database based on the columns of a pandas data frame.

    Parameters
    ----------
    df: pd.DataFrame
        The pandas dataframe object you wish to use the columns and data types
        to create the table from in postgresql
    database_name: str
        The name of the postgresql database.
    user: str
        The name of the user.
    table_name: str
        A string specifying the name of the newly created table.

    Returns
    -------
    str

    Examples
    --------
    >>> import seaborn as sns
    >>> from dotenv import load_dotenv, find_dotenv
    >>> from jmspack.internal_utils import add_data_to_postgresql_table
    >>> # Make sure you have a .env file somewhere with your postgresql credentials
    >>> # labelled as postgresql_host="BLA", and postgresql_password="BLA2"
    >>> load_dotenv(find_dotenv())
    >>> iris_df = sns.load_dataset("iris")
    >>> _ = create_postgresql_table_based_on_df(df=iris_df,
    ...                                         database_name="tracker",
    ...                                         user="tracker",
    ...                                         table_name="iris_test",
    ...                                         )

    """

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

    """Add new data to an existing table in a specified postgresql database.

    Parameters
    ----------
    df: pd.DataFrame
        The pandas dataframe object you wish to use the columns and data types
        to create the table from in postgresql
    database_name: str
        The name of the postgresql database.
    user: str
        The name of the user.
    table_name: str
        A string specifying the name of the newly created table.

    Returns
    -------
    str

    Examples
    --------
    >>> import seaborn as sns
    >>> from dotenv import load_dotenv, find_dotenv
    >>> from jmspack.internal_utils import add_data_to_postgresql_table
    >>> # Make sure you have a .env file somewhere with your postgresql credentials
    >>> # labelled as postgresql_host="BLA", and postgresql_password="BLA2"
    >>> load_dotenv(find_dotenv())
    >>> iris_df = sns.load_dataset("iris")
    >>> _ = create_postgresql_table_based_on_df(df=iris_df,
    ...                                         database_name="tracker",
    ...                                         user="tracker",
    ...                                         table_name="iris_test",
    ...                                         )
    >>> _ = add_data_to_postgresql_table(df=iris_df,
    ...                                 database_name="tracker",
    ...                                 user="tracker",
    ...                                 table_name="iris_test",
    ...                                 )

    """

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

    """Delete a table from a postgresql database.

    Parameters
    ----------
    database_name: str
        The name of the postgresql database.
    user: str
        The name of the user.
    table_name: str
        A string specifying the name of the newly created table.

    Returns
    -------
    str

    Examples
    --------
    >>> import seaborn as sns
    >>> from dotenv import load_dotenv, find_dotenv
    >>> from jmspack.internal_utils import add_data_to_postgresql_table
    >>> # Make sure you have a .env file somewhere with your postgresql credentials
    >>> # labelled as postgresql_host="BLA", and postgresql_password="BLA2"
    >>> load_dotenv(find_dotenv())
    >>> iris_df = sns.load_dataset("iris")
    >>> _ = create_postgresql_table_based_on_df(df=iris_df,
    ...                                         database_name="tracker",
    ...                                         user="tracker",
    ...                                         table_name="iris_test",
    ...                                         )
    >>> _ = add_data_to_postgresql_table(df=iris_df,
    ...                                 database_name="tracker",
    ...                                 user="tracker",
    ...                                 table_name="iris_test",
    ...                                 )
    >>> _ = delete_postgresql_table(database_name="tracker",
    ...                             user="tracker",
    ...                             table_name="iris_test"
    ...                             )

    """

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
