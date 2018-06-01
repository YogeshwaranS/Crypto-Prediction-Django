import os, shutil

print ("i am data script")


def my_func():
    print ("i am my func")


def reset_data_directory():
    cwd = os.getcwd()

    folder = cwd + '/Crypt/scripts'
    for the_file in os.listdir(folder):
        if the_file.endswith(tuple([".hdf5", ".csv", ".h5"])):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    print (file_path)
                    os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    csv_file = open(folder + "/data/bitcoin2015to2017.csv", "w")
    csv_file.close()


def download_data():
    reset_data_directory()
    print ("*** DOWNLOADING DATA FROM SCRIPT 1")
    import script_1
    print ("*** FILTERING DATA FROM SCRIPT 2")
    import Step_2_PastSamplerV1
    print ("*** CNN ML SCRIPT 3")
    import Step_3_CNN_V1


def get_graph_files():
    cwd = os.getcwd()
    folder = cwd + '/Crypt/scripts'

    files = os.listdir(folder)
    file_list = []
    for item in files:
        if item.endswith(".hdf5"):
            file_list.append(item)

    return file_list
