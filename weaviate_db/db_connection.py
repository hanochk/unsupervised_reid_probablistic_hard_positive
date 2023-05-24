import weaviate
import json
import sys
import os
sys.path.append(os.path.dirname(__file__))
# from face_rek import IMDBProcessor
import tqdm
import torch
import time
class WeaviateDB:

    def __init__(self):
        self.client = weaviate.Client("http://64.71.146.93:8080")
    
    def create_class_object(self):
        schema = self.client.schema.get()
        print(json.dumps(schema, indent=4))
        class_obj = {
            "class": "dataset_wiki_imdb", # <= Change to your class name - it will be your collection
            "description": "Dataset of celebrities",
            "vectorizer": "none",
            "properties": [
                            {
                            "dataType": [
                                "string"
                            ],
                            "description": "actor_name",
                            "name": "actor_name"
                        }
                    ]
        }
        self.client.schema.delete_class("dataset_wiki_imdb")
        self.client.schema.create_class(class_obj)

    def import_face_vectors(self, data):

        for idx, vector in enumerate(tqdm.tqdm(data.items())):
            try:
                actor_name = vector[0]
                embedding = vector[1]
                if embedding.shape == torch.Size([1, 512]):
                    self.client.data_object.create(
                            data_object={"actor_name": actor_name},
                            class_name='dataset_wiki_imdb',
                            vector=embedding
                        )
            except:
                print("Error")

    def search(self, vec="", limit=1):

        if vec != "":
            before = time.time()
            near_vec = {"vector": vec}
            res = self.client \
                .query.get("dataset_wiki_imdb", ["actor_name"]).with_additional('distance') \
                .with_near_vector(near_vec) \
                .with_limit(limit) \
                .do()
            search_took = time.time() - before
            data = res['data']['Get']['Dataset_wiki_imdb']
            print("Time it took to search: {:.4f}".format(search_took))

            return data

def main():
    db_instance = WeaviateDB()
    # db_instance.create_class_object()
    # imdb_instance = IMDBProcessor()
    # data_path = "/notebooks/nebula3_face_rekognition/actor_name_to_cluster.pkl"
    # data = imdb_instance.read_data(data_path, mode="pickle")
    # db_instance.import_face_vectors(data)


if __name__ == '__main__':
    main()