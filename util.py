def note_submission_info(msg, file_name):
    f = open("submission_history.txt", 'a')

    f.write(file_name + ":\n")
    f.write(msg)

    f.close()