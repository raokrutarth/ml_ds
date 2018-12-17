
from pymongo import MongoClient
from pprint import pprint
import datetime


class MongoDBTest:
    '''
        Setting up MongoDB on Ubuntu VM and running python client on Windows - 2 methods:

            Method 1) Installing MongoDB on host VM

            https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/
            https://www.mkyong.com/mongodb/mongodb-allow-remote-access/

                sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 9DA31620334BD75D9DCB49F368818C72E52529D4
                **The following instruction is for Ubuntu 16.04 (Xenial)
                echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/4.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
                sudo apt-get update
                sudo apt-get install -y mongodb-org
                sudo service mongod start
            ** on client
                pip install pymongo

            ** might have to create a user and a db on host terminal
                host-bash$ mongo
                > use TestDatabase
                > db.createUser(
                    {
                        user: "myUserAdmin",
                        pwd: "abc123",
                        roles: [ { role: "userAdminAnyDatabase", db: "admin" } ]
                    }
                )
            ** allow remote connections on DB port on host ip table
                iptables -A INPUT -p tcp --dport 27017 -j ACCEPT
            ** update mongodb config to allow remote connections
                vim /etc/mongod.conf

                    # Listen to local, LAN and Public interfaces.
                    bind_ip = 127.0.0.1,<desired allowed ip addresses>
            ** restart mongodb
                sudo service mongod restart

            Method 2) Get docker container on host VM

                host-vm$ docker pull mongo
                host-vm$ mkdir ~/data/
                host-vm$ docker run --name mongo_con -d -p <free port on host>:27017 -v ~/data:/data/ mongo
                ** verify with docker ps
                ** Connect as if the db service is running on port <free port on host>.
                ** the "-d" option in the docker command makes docker treat the container
                   as a daemon.

        Basic Python API:
            https://api.mongodb.com/python/current/tutorial.html

    '''
    port = '37017'
    host = '192.168.129.4'
    database  ='TestDatabase'
    def __init__(self):
        self.client = MongoClient('mongodb://{}:{}/'.format(
            self.host, self.port))
        print(self.client.list_database_names())
        self.db = self.client[self.database]

    def get_status(self):
        # Issue the serverStatus command and print the results
        pprint(self.db.command("serverStatus"))

    def insert_collection(self):
        car = {
            "name": "Toyota Yaris",
            "year": "2009",
            "tags": ["Small", "Hatchback", "1.5L"],
            "last_modified": datetime.datetime.now()
        }
        car_id = self.db.cars.insert_one(car).inserted_id
        pprint(car_id)

    def get_collections(self):
        res = self.db.collection_names(include_system_collections=False)
        pprint(res)

    def get_document(self):
        res = self.db.cars.find_one()
        pprint(res)

    def get_document_by_name(self):
        res = self.db.cars.find_one({"name": "Toyota Yaris"})
        pprint(res)

    def get_document_by_tag(self):
        res = self.db.cars.find_one({"tags": ["Small", "Hatchback", "1.5L"]})
        # for single element of the array:
        # res = self.db.cars.find_one({"tags": "1.5L"})
        pprint(res)

def main():
    mdb = MongoDBTest()
    # mdb.insert_collection()
    # mdb.get_collections()
    # mdb.get_document()
    # mdb.get_document_by_name()
    mdb.get_document_by_tag()
    return


if __name__ == '__main__':
    main()