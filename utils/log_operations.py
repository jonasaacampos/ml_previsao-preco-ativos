def write_log_in_txt(result_log, name_file):
    import os

    if not os.path.exists("results"):
        os.makedirs("results")

    with open("results" + os.sep + name_file + ".txt", "w") as file:
        for log in result_log:
            file.writelines(log + "\n")