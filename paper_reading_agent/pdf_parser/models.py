from pydantic import BaseModel, Field


class Section(BaseModel):
    title: str
    content: str
    level: int = 1
    start_page: int = 0
    end_page: int = 0
    parent_index: int | None = None


class Paper(BaseModel):
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    references: str = ""
    full_text: str = ""
    metadata: dict = Field(default_factory=dict)
