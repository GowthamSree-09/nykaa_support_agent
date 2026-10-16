import dataset


def test_dataset_valid():
    assert dataset.validate(dataset.ORDERS) == []


def test_dataset_reproducible():
    assert dataset.generate_orders() == dataset.generate_orders()
    assert dataset.generate_orders() == dataset.ORDERS


def test_fields_and_vocab():
    keys = {"record_id", "category", "status", "order_value_inr", "days_since_created", "delayed_shipment"}
    assert all(set(o) == keys for o in dataset.ORDERS)
    assert len(dataset.ORDERS) >= 40
