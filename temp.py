def receipt_to_text(parsed_json):
    """Convert Donut's parsed JSON output into a natural-language string for embedding."""
    menu = parsed_json.get('menu', [])
    if isinstance(menu, dict):
        menu = [menu]

    item_descriptions = []
    for item in menu:
        name = item.get('nm', 'unknown item')
        cnt = item.get('cnt', '')
        price = item.get('price', '')
        if cnt:
            item_descriptions.append(f"{cnt} x {name} for {price}")
        else:
            item_descriptions.append(f"{name} for {price}")

    items_text = ", ".join(item_descriptions)

    total = parsed_json.get('total', {})
    total_price = total.get('total_price', 'unknown')

    sentence = f"Receipt with items: {items_text}. Total amount: {total_price}."

    return sentence