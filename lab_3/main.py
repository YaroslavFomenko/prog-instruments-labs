import csv
import re
import json
from checksum import calculate_checksum


def validate_field(field_name: str, value: str) -> bool:
    """Проверяет валидность одного поля."""
    if not value:
        return False

    value = value.strip()

    if field_name == "telephone":
        return bool(re.match(r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$', value))

    elif field_name == "height":
        if not re.match(r'^\d\.\d{2}$', value):
            return False
        try:
            h = float(value)
            return 1.00 <= h <= 2.50
        except:
            return False

    elif field_name == "inn":
        return bool(re.match(r'^\d{12}$', value))

    elif field_name == "identifier":
        return bool(re.match(r'^\d{2}-\d{2}/\d{2}$', value))

    elif field_name == "occupation":
        if '_' in value:
            return False

        value_clean = value.strip()

        if not value_clean:
            return False

        if re.search(r'\d', value_clean):
            return False

        if not re.match(r'^[А-Яа-яЁёA-Za-z\s\-]+$', value_clean):
            return False

        if len(value_clean) < 2:
            return False

        if not re.search(r'[А-Яа-яЁёA-Za-z]', value_clean):
            return False

        if '  ' in value_clean:
            return False

        if value_clean.startswith('-') or value_clean.endswith('-'):
            return False

        if '--' in value_clean:
            return False

        words = [w for w in value_clean.split() if w]

        if len(words) < 1:
            return False

        for word in words:
            if not re.search(r'[А-Яа-яЁёA-Za-z]', word):
                return False

        for word in words:
            if word.replace('-', '') == '':
                return False

        return True

    elif field_name == "latitude":
        if '_' in value:
            return False
        if not re.match(r'^-?\d{1,2}\.\d+$', value):
            return False
        try:
            lat = float(value)
            return -90.0 <= lat <= 90.0
        except:
            return False

    elif field_name == "blood_type":
        if ' ' in value:
            return False

        value_norm = value.replace('−', '-')
        return bool(re.match(r'^(A|B|AB|O)[+\-]$', value_norm, re.IGNORECASE))

    elif field_name == "issn":
        return bool(re.match(r'^\d{4}-\d{4}$', value))

    elif field_name == "uuid":
        value_norm = value.strip('_').lower()

        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value_norm):
            return '_' not in value
        return False

    elif field_name == "date":
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return False

        try:
            year, month, day = map(int, value.split('-'))

            if year < 1900 or year > 2025:
                return False

            if month < 1 or month > 12:
                return False

            if day < 1 or day > 31:
                return False

            if month in [4, 6, 9, 11] and day > 30:
                return False

            if month == 2:
                if day > 29:
                    return False
                if day == 29 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                    return False

            return True

        except:
            return False

    return False


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

