import pyodbc

# Detaliile de conexiune
server = 'localhost'  # Portul 'localhost,1433'
database = 'Licenta'

# String-ul de conexiune
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# Realizarea conexiunii
try:
    conn = pyodbc.connect(connection_string)
    print("Conexiunea a fost realizată cu succes!")
except pyodbc.Error as ex:
    print("Eroare la conectare:")
    print(ex)
    exit(1)

# Crearea unui cursor
cursor = conn.cursor()

# Exemplu de interogare
cursor.execute('SELECT * FROM dbo.Users')

# Afișarea rezultatelor
for row in cursor.fetchall():
    print(row)

# Închiderea conexiunii
conn.close()
