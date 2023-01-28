import csv
from os.path import join as path_join

import dropbox
from twilio.rest import Client


def upload_to_dropbox(dropbox_access_token, filename, local_dropbox_dir, remote_dropbox_dir, instruments, weights):
    weights = [str(w) for w in weights]

    local_file = path_join(local_dropbox_dir, filename)

    with open(local_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(instruments)
        writer.writerow(weights)

    dbx = dropbox.Dropbox(dropbox_access_token)

    with open(local_file, 'rb') as file:
        dbx.files_upload(file.read(), '/' + remote_dropbox_dir + '/' + filename,
                         mode=dropbox.files.WriteMode('overwrite'))


def read_weights_dropbox(dropbox_access_token, filename):
    dbx = dropbox.Dropbox(dropbox_access_token)

    #   metadata, f = dbx.files_download('/policy/' + filename)

    metadata, f = dbx.files_download('/policy/' + filename)

    #   metadata, f = dbx.files_download(path_join('policy', filename))

    csv_reader = csv.reader(f.content.decode().splitlines(), delimiter=',')

    result = ''

    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #   print(row)
            instruments = row
            line_count += 1
        elif line_count == 1:
            #   print(row)
            suggested_weights = row
            line_count += 1

    return instruments, suggested_weights


def read_weights_file(filedir, filename):
    with open(path_join(filedir, filename), 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')

        #   csv_reader = csv.reader(f.content.decode().splitlines(), delimiter=',')

        result = ''

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #   print(row)
                instruments = row
                line_count += 1
            elif line_count == 1:
                #   print(row)
                suggested_weights = row
                line_count += 1

    return instruments, suggested_weights


def send_whatsup_message(message, media_url):
    client = Client('ACee2c8885b817a930407e963798454562', '40af3720220312bf212e344d973d9376')

    from_whatsapp_number = 'whatsapp:+14155238886'
    to_whatsapp_number = 'whatsapp:+972507774532'

    message = client.messages.create(
        body=message,
        media_url=media_url,
        from_=from_whatsapp_number,
        to=to_whatsapp_number)


#   https://www.geeksforgeeks.org/python-convert-a-list-to-dictionary/
def convert_list_to_dict(list):
    it = iter(list)
    res_dct = dict(zip(it, it))

    return res_dct


def convert_lists_to_dict(list1, list2):
    it1, it2 = iter(list1), iter(list2)
    res_dct = {k: v for k, v in zip(it1, it2)}

    return res_dct


def convert_dict_to_list(dict):
    v_dict_list = []
    for key, value in dict.items():
        v_key_value = [key, value]
        v_dict_list.extend(v_key_value)

    return v_dict_list


def get_name_from_dict_items(dict):
    name = ''
    for key, value in dict.items():
        name = name + key + '-' + str(value) + '-'

    return name[:-1]
