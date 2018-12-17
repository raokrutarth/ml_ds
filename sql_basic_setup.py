import pyodbc

class SQLTest:
    '''
        Installing Microsoft SQL server on Ubuntu on VirtualBox VM

        ** Setup host-only network on Virtual Box to expose VM on network

        https://www.microsoft.com/en-us/sql-server/developer-get-started/python/ubuntu/

        curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
        curl https://packages.microsoft.com/config/ubuntu/16.04/mssql-server-2017.list | sudo tee /etc/apt/sources.list.d/mssql-server-2017.list
        sudo apt-get update
        sudo apt-get install mssql-server
        sudo /opt/mssql/bin/mssql-conf setup
        sudo su
        curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
        curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
        exit
        sudo apt-get update
        sudo ACCEPT_EULA=Y apt-get install msodbcsql17 mssql-tools
        echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile
        echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
        source ~/.bashrc
        sudo apt-get install unixodbc-dev
        sqlcmd -S localhost -U sa -P <password> -Q "SELECT @@VERSION"
        pip install pyodbc

        **Install Microsoft ODBC Driver 17 for SQL Server on client

    '''
    server = '192.168.129.4' # VM ip address
    database = 'TestDB'
    username = 'sa'
    password = 'Admin123'
    connection = pyodbc.connect(
                    'DRIVER={ODBC Driver 17 for SQL Server};' +
                    'SERVER='+ server +
                    ';PORT=1443;' +
                    'DATABASE=' + database +
                    ';UID=' + username +
                    ';PWD=' + password)
    # need to set to true if a db needs to
    # be created. i.e. auto single command
    connection.autocommit = True
    cursor = connection.cursor()
    table = 'SpeedMap'

    def create_db(self):
        self.cursor.execute('CREATE DATABASE {}'.format(self.database))

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE {}
            (id int IDENTITY(1,1) PRIMARY KEY,
            location varchar(100) NULL,
            speed BIGINT NULL,
            creationTime datetime default CURRENT_TIMESTAMP)
            """.format(self.table))
        print('Rows modified: ', self.cursor.rowcount)

    def insert(self, location, speed):
        tx = 'INSERT INTO {} (location, speed) VALUES (?, ?)'.format(self.table)
        self.cursor.execute(tx, location, speed)
        print('[Insert] Rows modified: ', self.cursor.rowcount)

    def update_speed(self, location, new_speed):
        #Update Query
        tsql = "UPDATE {} SET speed = ? WHERE location = ?".format(self.table)
        self.cursor.execute(tsql, new_speed, location)
        print('[Update] Rows modified: ', self.cursor.rowcount)

    def delete(self, location):
        #Delete Query
        tsql = "DELETE FROM {} WHERE location = ?".format(self.table)
        self.cursor.execute(tsql, location)
        print('[Update] Rows modified: ', self.cursor.rowcount)

    def select_all(self):
        #Select Query
        print ('Reading data from table')
        tsql = "SELECT location, speed FROM SpeedMap;"
        with self.cursor.execute(tsql):
            row = self.cursor.fetchone()
            while row:
                print (str(row[0]) + " " + str(row[1]))
                row = self.cursor.fetchone()

    def close(self):
        self.connection.close()


def main():
    sql_test = SQLTest()
    # sql_test.create_db()
    # sql_test.create_table()
    # sql_test.insert2()
    sql_test.select_all()
    sql_test.update_speed('Ghana', '878')
    sql_test.select_all()
    sql_test.close()


if __name__ == '__main__':
    main()