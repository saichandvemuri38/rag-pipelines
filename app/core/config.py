from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
