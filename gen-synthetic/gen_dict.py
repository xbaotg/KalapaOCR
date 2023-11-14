import pickle
import requests


country = pickle.load(
    open(
        "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/postprocess/fix-address/data/country_unique.pkl",
        "rb",
    )
)
total_address = list(country["TTP"]) + list(country["QH"]) + list(country["PX"])

with open("resources/vi.txt", "w+") as f:
    allowed_char = (
        "".join(
            [
                f.strip()
                for f in open(
                    "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/labels.txt"
                ).readlines()
            ]
        )
        + " "
    )

    for address in total_address:
        if address is None or address == "" or str(address) == "nan":
            continue

        available = True
        for y in address:
            if y not in allowed_char:
                available = False
                break

        if not available:
            continue

        f.write(address + "\n")

    street = set()
    response = requests.get(
        "https://cdn.jsdelivr.net/gh/thien0291/vietnam_dataset@1.0.0/Index.json"
    )

    for i, (k, v) in enumerate(response.json().items()):
        code = v["code"]
        res = requests.get(
            f"https://cdn.jsdelivr.net/gh/thien0291/vietnam_dataset@1.0.0/data/{code}.json"
        ).json()

        for t in res["district"]:
            for z in t["street"]:
                if len(z) < 2:
                    continue

                available = True
                for y in z:
                    if y not in allowed_char:
                        available = False
                        break

                if available:
                    street.add(z)

        # print(i, len(street))

    for s in street:
        f.write(s + "\n")
