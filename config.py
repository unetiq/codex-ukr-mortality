import configparser
import ast

config = configparser.ConfigParser()
config.read("config.ini")
config = config["FEATURES"]

ICU_FEATURES = ast.literal_eval(config.get("ICU_FEATURES"))
