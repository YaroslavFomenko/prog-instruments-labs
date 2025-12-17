import csv
import re
import json
from checksum import calculate_checksum


def validate_field(field_name: str, value: str) -> bool:
    """Проверяет валидность одного поля регулярными выражениями."""
    if not value:
        return False

    value = value.strip()

    patterns = {
        "telephone": r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$',

        "height": r'^(?:1\.[0-9]{2}|2\.[0-4][0-9]|2\.50)$',

        "inn": r'^\d{12}$',

        "identifier": r'^\d{2}-\d{2}/\d{2}$',

        "occupation": r'^(?=[А-Яа-яЁёA-Za-z])[А-Яа-яЁёA-Za-z](?:(?![_\d]|--|  )[А-Яа-яЁёA-Za-z\s\-])*(?<=[А-Яа-яЁёA-Za-z])$',

        "latitude": r'^(?:-?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?))$',

        "blood_type": r'^(?:A|B|AB|O)[+\u2212\-]$',

        "issn": r'^\d{4}-\d{4}$',

        "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',

        "date": r'^((?:19\d{2}|20[0-2]\d|202[0-5])-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])|(?:19|20)(?:[02468][048]|[13579][26])-02-29)$'
    }

    if field_name not in patterns:
        return False

    if field_name == "blood_type":
        value = value.replace('\u2212', '-').upper()
    elif field_name == "uuid":
        if '_' in value:
            return False

        value_norm = value.strip('_').lower()

        if not re.match(patterns[field_name], value_norm):
            return False

        return True

    return bool(re.match(patterns[field_name], value))


def find_invalid_rows(file_path: str):
    """Находит все невалидные строки."""
    invalid_rows = []

    try:
        with open(file_path, 'r', encoding='utf-16') as file:
            csv_reader = csv.reader(file, delimiter=';', quotechar='"')

            header = next(csv_reader)
            header = [h.strip('"').strip() for h in header]

            field_indices = {}
            for field in ["telephone", "height", "inn", "identifier", "occupation",
                          "latitude", "blood_type", "issn", "uuid", "date"]:
                field_indices[field] = header.index(field)

            row_num = 0
            for row in csv_reader:
                for field in field_indices:
                    idx = field_indices[field]
                    if idx < len(row):
                        value = row[idx].strip().strip('"')
                        if not validate_field(field, value):
                            invalid_rows.append(row_num)
                            break

                row_num += 1

    except Exception as e:
        print(f"Ошибка: {e}")

    return invalid_rows


# Запускаем
invalid_rows = find_invalid_rows("24.csv")
print(f"Найдено невалидных строк: {len(invalid_rows)}")

# Проверяем чексумму
checksum = calculate_checksum(invalid_rows)
print(f"\nКонтрольная сумма: {checksum}")

# Сохраняем
result = {"variant": 24, "checksum": checksum}
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("Результат записан в result.json")

# Подсчет ошибок по всем полям
all_fields = ["telephone", "height", "inn", "identifier", "occupation",
              "latitude", "blood_type", "issn", "uuid", "date"]
error_counts = {field: 0 for field in all_fields}

try:
    with open("24.csv", 'r', encoding='utf-16') as file:
        csv_reader = csv.reader(file, delimiter=';', quotechar='"')

        header = next(csv_reader)
        header = [h.strip('"').strip() for h in header]

        field_indices = {}
        for field in all_fields:
            field_indices[field] = header.index(field)

        row_num = 0
        for row in csv_reader:
            for field in all_fields:
                idx = field_indices[field]
                if idx < len(row):
                    value = row[idx].strip().strip('"')
                    if not validate_field(field, value):
                        error_counts[field] += 1

            row_num += 1

except Exception as e:
    print(f"Ошибка при подсчете: {e}")

print("\nСтатистика ошибок по полям:")
for field, count in error_counts.items():
    if count > 0:
        print(f"{field}: {count} ошибок")