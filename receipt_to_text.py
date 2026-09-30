def receipt_to_text(parsed_json):
    menu = parsed_json.get('menu', [])
    if isinstance(menu, dict):
        menu = [menu]

    item_descriptions = []
    for item in menu:
        name = item.get('nm', 'unknown item')
        cnt = item.get('cnt', '')
        price = item.get('price', '')
        if cnt:
            item_descriptions.append(f"{cnt} {name} for {price}")
        else:
            item_descriptions.append(f"{name} for {price}")

    items_text = ", ".join(item_descriptions)

    total = parsed_json.get('total', {})
    total_price = total.get('total_price', 'unknown')

    sentence = f"Receipt with items: {items_text}. Total amount: {total_price}."

    return sentence


if __name__ == "__main__":
    sample_json = {
        "menu": [
            {"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"},
            {"nm": "EGG TART", "cnt": "1", "price": "13,000"}
        ],
        "total": {"total_price": "45,500"}
    }
    print(receipt_to_text(sample_json))