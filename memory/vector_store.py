URI = "http://localhost:19530"

from langchain_milvus import Milvus
from utils.helpers import embeddings

### Milvus Vector DB Creating if it does not exist


from pymilvus import Collection, MilvusException, connections, db, utility

conn = connections.connect(host="127.0.0.1", port=19530)

db_name = "milvus_assignment_test"
try:
    existing_databases = db.list_database()
    db_was_deleted = False

    if db_name in existing_databases:
        print(f"Database '{db_name}' already exists.")

        # Use the database context
        db.using_database(db_name)

        # Drop all collections in the database
        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Collection '{collection_name}' has been dropped.")

        # Drop the database
        db.drop_database(db_name)
        db_was_deleted = True
        print(f"Database '{db_name}' has been deleted.")
    else:
        print(f"Database '{db_name}' does not exist.")
        db_was_deleted = True

    # Recreate the database if it was deleted
    if db_was_deleted:
        db.create_database(db_name)
        print(f"Database '{db_name}' created successfully.")

except MilvusException as e:
    print(f"An error occurred: {e}")


#### Milvus DB Local Instance running on Docker- connected below

# Creating a FLAT index vector store with L2 similarity check


flat_milvus_vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "milvus_assignment_test"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    collection_name="Flat_Index_Used_Car_PDF_Collection",
    collection_description= "Its the contents from the PDF, explaining how to buy used cars. It uses Flat Index.",
    consistency_level="Strong"

)


# Web Crawler data's vector store

web_crawler_milvus_vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "milvus_assignment_test"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    collection_name="Flat_Index_Web_Crawler_Data",
    collection_description= "Its the website scraped contents from a used car dealership's inventory.",
    consistency_level="Strong"

)

print("\nEmpty vector stores has been created successfully")