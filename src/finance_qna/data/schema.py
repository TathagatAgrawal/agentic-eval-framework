"""SQLAlchemy Core table definitions for the synthetic transaction database."""

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
)

metadata = MetaData()

accounts = Table(
    "accounts",
    metadata,
    Column("account_id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("type", String, nullable=False),
)

categories = Table(
    "categories",
    metadata,
    Column("category_id", Integer, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("parent_category_id", Integer, ForeignKey("categories.category_id"), nullable=True),
)

merchants = Table(
    "merchants",
    metadata,
    Column("merchant_id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("default_category_id", Integer, ForeignKey("categories.category_id"), nullable=False),
)

transactions = Table(
    "transactions",
    metadata,
    Column("transaction_id", Integer, primary_key=True),
    Column("account_id", Integer, ForeignKey("accounts.account_id"), nullable=False),
    Column("merchant_id", Integer, ForeignKey("merchants.merchant_id"), nullable=False),
    Column("category_id", Integer, ForeignKey("categories.category_id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("amount", Numeric(12, 2), nullable=False),
    Column("description", String, nullable=False),
    Column("is_subscription", Boolean, nullable=False, default=False),
)
