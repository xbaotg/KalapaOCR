import pandas as pd
import pickle
import time
import argparse
from Levenshtein import distance as levenshtein_distance
from thefuzz import fuzz
from unidecode import unidecode
from fuzzysearch import find_near_matches


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="")
parser.add_argument("--output", type=str, default="processed_result.csv")
args = parser.parse_args()

print("Loading data...")
df = pd.read_csv(args.input)
dict_country_no_prefix = pickle.load(open("data/country_map_no_prefix.pkl", "rb"))
street_data = pickle.load(open("data/street.pkl", "rb"))
print("Done.")


NGANG = "u u e e o o o a a a i y"  # 0
SAC = "ú ứ é ế ó ố ớ á ắ ấ í ý"  # 1
HUYEN = "ù ừ è ề ò ồ ờ à ằ ầ ì ỳ"  # 2
HOI = "ủ ử ẻ ể ỏ ổ ở ả ẳ ẩ ỉ ỷ"  # 3
NGA = "ũ ữ ẽ ễ õ ỗ ỡ ã ẵ ẫ ĩ ỹ"  # 4
NANG = "ụ ự ẹ ệ ọ ộ ợ ạ ặ ậ ị ỵ"  # 5


def get_type(ch):
    if ch in NGANG:
        return 0
    if ch in SAC:
        return 1
    elif ch in HUYEN:
        return 2
    elif ch in HOI:
        return 3
    elif ch in NGA:
        return 4
    elif ch in NANG:
        return 5
    else:
        return -1


def change_tone(ch, type):
    idx = -1

    for t in [SAC, HUYEN, HOI, NGA, NANG, NGANG]:
        if ch in t:
            idx = t.index(ch)

    if type == 0:
        return NGANG[idx]
    elif type == 1:
        return SAC[idx]
    elif type == 2:
        return HUYEN[idx]
    elif type == 3:
        return HOI[idx]
    elif type == 4:
        return NGA[idx]
    elif type == 4:
        return NANG[idx]
    else:
        return ch


# generate ignore list
ignore_chars = "a,á,à,ạ,ả,ã,ă,ắ,ằ,ặ,ẳ,ẵ,â,ấ,ầ,ậ,ẩ,ẫ,o,ó,ò,ọ,ỏ,õ,ơ,ớ,ờ,ợ,ở,ỡ,ô,ố,ồ,ộ,ổ,ỗ,u,ú,ù,ụ,ủ,ũ,ư,ứ,ừ,ự,ử,ữ,e,é,è,ẹ,ẻ,ẽ,ê,ế,ề,ệ,ể,ễ,i,í,ì,ị,ỉ,ĩ,y,ý,ỳ,ỵ,ỷ,ỹ"
ignore = []

for c in ignore_chars.split(","):
    for c2 in ignore_chars.split(","):
        if c != c2:
            ignore.append(
                (c + c2, change_tone(c, get_type(c2)) + change_tone(c2, get_type(c)))
            )


def main():
    cnt = 0

    output = open(args.output, "w+")
    output.write("id,answer\n")

    for row in df.itertuples():
        id_query = row.id

        if isinstance(row.answer, str):
            ori_text = row.answer
        else:
            ori_text = ""

        # if id_query != "164/0.jpg":
        #     continue
        
        text = unidecode(ori_text.lower())
        possible_country = {}
        total_score = {}
        last_update_score = 1000
        allowed_diff = 0.5
        diff_between_edit = 4

        for ori_k in dict_country_no_prefix.keys():
            k = unidecode(ori_k.lower())

            split_text = text.split(" ")
            split_k = k.split(" ")

            cur_text = " ".join(split_text[-len(split_k) :])
            allowed_score = max(1, len(k) // diff_between_edit)
            len_country = len(k.split(" "))
            cur_score = levenshtein_distance(k, cur_text) / allowed_score

            if cur_score <= 1:
                matched_country = ori_k
                matched_text = " ".join(ori_text.split(" ")[-len_country:])

                if (last_update_score - cur_score) > allowed_diff:
                    possible_country = {}
                    last_update_score = cur_score
                    # print("Update score: ", last_update_score, matched_country, matched_text)

                if abs(last_update_score - cur_score) <= allowed_diff:
                    possible_country[matched_country] = {
                        "info": (
                            matched_text,
                            cur_score,
                            len(text) - len(matched_text),
                            len(text),
                        )
                    }

        # TODO: Thanh ho'a , Thanh hoa' => Can check xem neu bo dau ma khac nhau thi moi can replace

        # Thanh pho / Thi xa
        last_update_score = 1000

        if len(possible_country.keys()):
            for country, info_country in possible_country.items():
                total_score[country] = 1

                for ori_k in dict_country_no_prefix[country].keys():
                    k = unidecode(ori_k.lower())
                    split_k = k.split(" ")
                    split_text = text.split(" ")

                    for i in range(len(text) - len(split_k) + 1):
                        # index of last character
                        start_position = len(" ".join(split_text[:i])) + (
                            1 if i > 0 else 0
                        )
                        end_position = len(" ".join(split_text[: i + len(split_k)])) - 1

                        # overlap ten tinh / thanh pho
                        if end_position >= info_country["info"][2]:
                            break

                        cur_text = " ".join(split_text[i : i + len(split_k)])

                        if any(t in cur_text for t in "0123456789"):
                            continue

                        cur_score = levenshtein_distance(k, cur_text) / max(
                            1, len(k) // diff_between_edit
                        )

                        if cur_score <= 1:
                            matched_text = " ".join(
                                ori_text.split(" ")[i : i + len(split_k)]
                            )

                            if (last_update_score - cur_score) > allowed_diff:
                                possible_country[country]["match"] = {}
                                last_update_score = cur_score

                            if abs(last_update_score - cur_score) <= allowed_diff:
                                if "match" not in possible_country[country]:
                                    possible_country[country]["match"] = {}

                                possible_country[country]["match"][ori_k] = {
                                    "info": (
                                        matched_text,
                                        cur_score,
                                        start_position,
                                        start_position + len(matched_text),
                                    )
                                }

        # print(possible_country)

        last_update_score = 1000
        if len(possible_country.keys()):
            for country, info_country in possible_country.items():
                if "match" not in info_country:
                    continue

                for city, info_city in possible_country[country]["match"].items():
                    for ori_k in dict_country_no_prefix[country][city]:
                        if str(ori_k) == "nan":
                            continue

                        k = unidecode(ori_k.lower())
                        split_k = k.split(" ")
                        split_text = text.split(" ")

                        for i in range(len(text) - len(split_k) + 1):
                            # index of last character
                            # abc affa
                            start_position = len(" ".join(split_text[:i])) + (
                                1 if i > 0 else 0
                            )
                            end_position = (
                                len(" ".join(split_text[: i + len(split_k)])) - 1
                            )

                            # overlap ten tinh / thanh pho
                            if end_position >= info_city["info"][2]:
                                break

                            cur_text = " ".join(split_text[i : i + len(split_k)])

                            if any(t in cur_text for t in "0123456789"):
                                continue

                            cur_score = levenshtein_distance(k, cur_text) / max(
                                1, len(k) // diff_between_edit
                            )

                            if cur_score <= 1:
                                matched_text = " ".join(
                                    ori_text.split(" ")[i : i + len(split_k)]
                                )

                                if (last_update_score - cur_score) > allowed_diff:
                                    # print("Update score: ", country, city)
                                    possible_country[country]["match"][city][
                                        "match"
                                    ] = {}

                                    last_update_score = cur_score
                                    # print("Update score: ", last_update_score, matched_country, matched_text)

                                if abs(last_update_score - cur_score) <= allowed_diff:
                                    if (
                                        "match"
                                        not in possible_country[country]["match"][city]
                                    ):
                                        possible_country[country]["match"][city][
                                            "match"
                                        ] = {}

                                    possible_country[country]["match"][city]["match"][
                                        ori_k
                                    ] = {
                                        "info": (
                                            matched_text,
                                            cur_score,
                                            start_position,
                                            start_position + len(matched_text),
                                        )
                                    }

        # street
        last_update_score = 1000
        diff_between_edit = 6

        for k in possible_country.keys():
            if "match" in possible_country[k]:
                for city in possible_country[k]["match"].keys():
                    # ten phuong / xa
                    if "match" in possible_country[k]['match'][city].keys():
                        print("Match: ", k, city)
                        for phuong in possible_country[k]['match'][city]['match']:
                            left_pos = possible_country[k]['match'][city]['match'][phuong]['info'][2]

                            if k in street_data.keys() and city in street_data[k].keys():
                                for p_street in street_data[k][city]:
                                    t_k = unidecode(p_street.lower())
                                    split_k = t_k.split(" ")
                                    split_text = text.split(" ")

                                    for i in range(len(text) - len(split_k) + 1):
                                        # index of last character
                                        # abc affa
                                        start_position = len(" ".join(split_text[:i])) + (
                                            1 if i > 0 else 0
                                        )
                                        end_position = (
                                            len(" ".join(split_text[: i + len(split_k)])) - 1
                                        )

                                        # overlap ten tinh / thanh pho
                                        if end_position >= left_pos:
                                            break

                                        cur_text = " ".join(split_text[i : i + len(split_k)])

                                        if any(t in cur_text for t in "0123456789"):
                                            continue

                                        cur_score = levenshtein_distance(t_k, cur_text) / max(
                                            1, len(t_k) // diff_between_edit
                                        )

                                        if cur_score <= 1:
                                            matched_text = " ".join(
                                                ori_text.split(" ")[i : i + len(split_k)]
                                            )

                                            if (last_update_score - cur_score) > allowed_diff:
                                                # print("Update score: ", country, city)
                                                possible_country[k]["match"][city]["match"][phuong]['match'] = {}
                                                last_update_score = cur_score
                                                # print("Update score: ", last_update_score, matched_country, matched_text)

                                            if abs(last_update_score - cur_score) <= allowed_diff:
                                                if (
                                                    "match"
                                                    not in possible_country[k]["match"][city]['match'][phuong]
                                                ):
                                                    possible_country[k]["match"][city]["match"][phuong]['match'] = {}

                                                possible_country[k]["match"][city]['match'][phuong]["match"][
                                                    p_street
                                                ] = {
                                                    "info": (
                                                        matched_text,
                                                        cur_score,
                                                        start_position,
                                                        start_position + len(matched_text),
                                                    )
                                                }

                                            # input()
                            else:
                                pass
                    else:
                        left_pos = possible_country[k]['match'][city]['info'][2]

                        if k in street_data.keys() and city in street_data[k].keys():
                            for p_street in street_data[k][city]:
                                t_k = unidecode(p_street.lower())
                                split_k = t_k.split(" ")
                                split_text = text.split(" ")

                                for i in range(len(text) - len(split_k) + 1):
                                    # index of last character
                                    # abc affa
                                    start_position = len(" ".join(split_text[:i])) + (
                                        1 if i > 0 else 0
                                    )
                                    end_position = (
                                        len(" ".join(split_text[: i + len(split_k)])) - 1
                                    )

                                    # overlap ten tinh / thanh pho
                                    if end_position >= left_pos:
                                        break

                                    cur_text = " ".join(split_text[i : i + len(split_k)])

                                    if any(t in cur_text for t in "0123456789"):
                                        continue

                                    cur_score = levenshtein_distance(t_k, cur_text) / max(
                                        1, len(t_k) // diff_between_edit
                                    )

                                    if cur_score <= 1:
                                        matched_text = " ".join(
                                            ori_text.split(" ")[i : i + len(split_k)]
                                        )
                                        print("Found: ", p_street, city, phuong, matched_text)

                                        if (last_update_score - cur_score) > allowed_diff:
                                            # print("Update score: ", country, city)
                                            possible_country[k]["match"][city]['match'] = {}

                                            last_update_score = cur_score
                                            # print("Update score: ", last_update_score, matched_country, matched_text)

                                        if abs(last_update_score - cur_score) <= allowed_diff:
                                            if (
                                                "match"
                                                not in possible_country[k]["match"][city]
                                            ):
                                                possible_country[k]["match"][city]['match'] = {}

                                            possible_country[k]["match"][city]['match'][
                                                p_street
                                            ] = {
                                                "info": (
                                                    matched_text,
                                                    cur_score,
                                                    start_position,
                                                    start_position + len(matched_text),
                                                )
                                            }

        print(possible_country)
        results = []

        for k in possible_country.keys():
            if "match" in possible_country[k]:
                for city in possible_country[k]["match"].keys():
                    if "match" in possible_country[k]["match"][city]:
                        for street in possible_country[k]["match"][city][
                            "match"
                        ].keys():
                            if "match" in possible_country[k]["match"][city]["match"][street]:
                                for p_street in possible_country[k]["match"][city][
                                    "match"
                                ][street]["match"].keys():
                                    results.append(
                                        (
                                            4 + 1000 
                                            - possible_country[k]["info"][1]
                                            - possible_country[k]["match"][city]["info"][1]
                                            - possible_country[k]["match"][city]["match"][street]['info'][1]
                                            - possible_country[k]["match"][city]["match"][street]['match'][p_street]['info'][1],
                                            1000
                                            - possible_country[k]["info"][1]
                                            - possible_country[k]["match"][city]["info"][1]
                                            - possible_country[k]["match"][city]["match"][ street ]["info"][1]
                                            - possible_country[k]["match"][city]["match"][street]['match'][p_street]['info'][1],
                                            (
                                                (k,) + possible_country[k]["info"],
                                                (city,) + possible_country[k]["match"][city]["info"],
                                                (street,) + possible_country[k]["match"][city]["match"][ street ]["info"],
                                                (p_street,) + possible_country[k]["match"][city]["match"][street]['match'][p_street]["info"],
                                            ),
                                        )
                                    )
                            else:
                                results.append(
                                    (
                                        3 + 1000 
                                        - possible_country[k]["info"][1]
                                        - possible_country[k]["match"][city]["info"][1]
                                        - possible_country[k]["match"][city]["match"][
                                            street
                                        ]['info'][1],
                                        1000
                                        - possible_country[k]["info"][1]
                                        - possible_country[k]["match"][city]["info"][1]
                                        - possible_country[k]["match"][city]["match"][
                                            street
                                        ]["info"][1],
                                        (
                                            (k,) + possible_country[k]["info"],
                                            (city,)
                                            + possible_country[k]["match"][city]["info"],
                                            (street,)
                                            + possible_country[k]["match"][city]["match"][
                                                street
                                            ]["info"],
                                        ),
                                    )
                                )
                    else:
                        results.append(
                            (
                                2 + 
                                1000
                                - possible_country[k]["info"][1]
                                - possible_country[k]["match"][city]["info"][1],
                                1000
                                - possible_country[k]["info"][1]
                                - possible_country[k]["match"][city]["info"][1],
                                (
                                    (k,) + possible_country[k]["info"],
                                    (city,)
                                    + possible_country[k]["match"][city]["info"],
                                ),
                            )
                        )
            else:
                results.append(
                    (
                        1 + 
                        1000 - possible_country[k]["info"][1],
                        1000 - possible_country[k]["info"][1],
                        ((k,) + possible_country[k]["info"],),
                    )
                )

        results.sort(reverse=True)
        print(results)

        if len(results):
            choice = results[0]

            if len(results) > 1:
                last_distance = 1000
                last_total_pos = 0

                for i in range(len(results) - 1):
                    if (
                        results[i][1] == results[i + 1][1]
                        and results[i][0] == results[i + 1][0]
                    ):
                        m = results[i][2]
                        n = results[i + 1][2]

                        score_m = 0
                        score_n = 0

                        for t in m:
                            score_m += levenshtein_distance(t[0], t[1])

                        for t in n:
                            score_n += levenshtein_distance(t[0], t[1])

                        if score_m <= last_distance:
                            total_pos = sum(t[3] for t in m)

                            if score_m < last_distance or total_pos > last_total_pos:
                                choice = results[i]
                                last_distance = score_m
                                last_total_pos = total_pos

                        if score_n <= last_distance:
                            total_pos = sum(t[3] for t in n)

                            if score_n < last_distance or total_pos > last_total_pos:
                                choice = results[i + 1]
                                last_distance = score_n
                                last_total_pos = total_pos

                        # print("-----------------")
                        # print(results)
                        # print(choice)
                    else:
                        break

            t = choice
            # print(t)

            need_replace = []
            # print(t)
            for z in t[2]:
                matched_text = ori_text[z[3] : z[4]]
                replace_text = z[0]

                matched_text_uni = text[z[3] : z[4]]
                replace_text_uni = unidecode(z[0].lower())

                matched_text_uni_split = matched_text_uni.split(" ")
                replace_text_uni_split = replace_text_uni.split(" ")
                matched_text_split = matched_text.split(" ")
                replace_text_split = replace_text.split(" ")

                # if len(matched_text_uni_split) == len(replace_text_uni_split):
                #     for i in range(len(matched_text_uni_split)):
                #         if matched_text_uni_split[i] == replace_text_uni_split[i]:
                #             for k in ignore:
                #                 if k[0] in matched_text_uni_split[i]:
                #                     print("K: ", k, matched_text_uni_split[i], replace_text_uni_split[i])
                #                     replace_text_split[i] = matched_text_split[i]
                #                     break

                if len(matched_text_uni_split) == len(replace_text_uni_split):
                    for i in range(len(matched_text_uni_split)):
                        if matched_text_uni_split[i] == replace_text_uni_split[i]:
                            for k in ignore:
                                if (
                                    k[0] in matched_text_split[i]
                                    and k[1] in replace_text_split[i]
                                ):
                                    replace_text_split[i] = matched_text_split[i]
                                    break

                replace_text = " ".join(replace_text_split)
                need_replace.append((matched_text, replace_text))

                print(
                    "Replace: ",
                    id_query,
                    "\t",
                    matched_text,
                    "\t",
                    replace_text,
                    "\t",
                    ori_text,
                )

        else:
            print("Not found: ", id_query, "\t", ori_text)

        for t in need_replace:
            ori_text = ori_text.replace(t[0], t[1])

        # ori_text = ori_text.replace("óa")"oá",
        label = ori_text.split(" ")
        for idx in range(len(label)):
            if idx == 0:
                continue

            if len(label[idx-1]) == 1 and label[idx].isnumeric() and not label[idx-1].isnumeric():
                label[idx-1] = label[idx-1] + label[idx]
                label[idx] = ""
                continue

        ori_text = " ".join(label)
        ori_text = ori_text.replace("  ", " ")
        if "Thủy" in ori_text: 
            ori_text = ori_text.replace("Thủy","Thuỷ")
        if "Hòa"in ori_text:
            ori_text = ori_text.replace("Hòa","Hoà")
            

        # print(ori_text)
        output.write(f"{id_query},{ori_text}\n")

        # input()

    print("cnt = ", cnt)
    output.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total time: ", (time.time() - start_time) / len(list(df.itertuples())))
