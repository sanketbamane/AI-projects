from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models import Base
config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata
def run_migrations_offline():
    pass
def run_migrations_online():
    pass
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()