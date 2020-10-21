import glob


def valid_credentials(api_key):
    lines = open("api_keys.txt", "r").readlines()
    for line in lines:
        username, key = line.strip().split(":")
        if key == api_key: return True
    return False


def get_username(api_key):
    lines = open("api_keys.txt", "r").readlines()
    for line in lines:
        username, key = line.strip().split(":")
        if key == api_key: return username
    return None


def get_model_types():
    available_models = [f for f in glob.glob("models/*")]
    return [m.split("/")[-1] for m in available_models]


def get_models(model_type):
    available_models = [f for f in glob.glob(f"models/{model_type}/*.hdf5")]
    return sorted([m.split("/")[-1].split(".")[0] for m in available_models])


def merge_model(model_type, new_model_path):
    latest_model = get_models(model_type)[-1]
    # todo: 
    # - evaluate new model. Only continue if model is above certain dice threshold
    # - merge latest_model and new_model_path 
    # - evaluate merged model
    # - store new model if it is above certain dice threshold


if __name__ == '__main__':
    # valid_credentials("abc123")
    # print(get_model_types())
    print(get_models("thigh"))

