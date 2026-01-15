# ADR 0022: Use Pydantic `from_attributes` for UI API response schemas

Date: 2026-01-15

Context: UI API routers duplicated large amounts of manual field copying (getattr/list/dict fallbacks), increasing code size and creating inconsistent defaults across endpoints.
Decision: Introduce a shared `OrmOutModel` base with `model_config = ConfigDict(from_attributes=True)` and move default/coercion logic (e.g. Enum->str, None->[]/{}, display fallbacks) into schema validators.
Constraints: Avoid triggering SQLAlchemy lazy loads by validating only column-like attributes from ORM objects; build composite responses (e.g. commit detail metrics/artifacts) explicitly rather than relying on ORM relationships.
Consequences: Routers can return ORM/dataclass objects directly, reducing boilerplate and centralizing response semantics in schemas; validation becomes more consistent and easier to evolve.

