import os

microrts_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(microrts_path, "data")
models_dir = os.path.join(microrts_path, "models")
jar_dir = os.path.join(microrts_path, "rts_wrapper/microrts-java.jar")
map_dir = os.path.join(microrts_path, "maps")

client_ip = "127.0.0.1"

# print(microrts_path)