"""Gaia Predictive — Base Service Interface.

Implements the Service Repository pattern to decouple business logic
from API routes. All services inherit from this base class.

Features:
    - Generic CRUD operations (get, create, update, delete)
    - Automatic logging with context
    - Standardized error handling
    - Type-safe session management

Usage:
    class MachineService(BaseService[Machine, MachineCreate, MachineUpdate]):
        def __init__(self, db: AsyncSession):
            super().__init__(Machine, db)
            
        async def custom_logic(self):
            ...
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, cast
import uuid

from fastapi import HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from logger import get_logger

# Generics for strong typing
ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

logger = get_logger(__name__)


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base class for all business logic services.
    
    Provides standard CRUD operations and session management.
    API routes should use these services instead of raw DB usage.
    """
    
    def __init__(self, model: type[ModelType], db: AsyncSession):
        """Initialize service with model class and database session.
        
        Args:
            model: The SQLAlchemy model class.
            db: The async database session.
        """
        self.model = model
        self.db = db
        self.logger = logger.bind(service=self.__class__.__name__)
    
    async def get(self, id: Any) -> ModelType | None:
        """Get a single record by primary key."""
        result = await self.db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100
    ) -> list[ModelType]:
        """Get multiple records with pagination."""
        result = await self.db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())
    
    async def create(self, obj_in: CreateSchemaType) -> ModelType:
        """Create a new record."""
        # Convert Pydantic model to dict
        obj_in_data = obj_in.model_dump()
        
        # Instantiate model
        db_obj = self.model(**obj_in_data)  # type: ignore
        
        self.db.add(db_obj)
        try:
            await self.db.commit()
            await self.db.refresh(db_obj)
            
            self.logger.info(
                "Created new record",
                id=str(getattr(db_obj, "id", None)),
                model=self.model.__name__
            )
            return db_obj
            
        except IntegrityError as e:
            await self.db.rollback()
            self.logger.warning(
                "Create failed - integrity error",
                error=str(e),
                data=obj_in_data
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Resource already exists or violates constraints"
            ) from e
    
    async def update(
        self,
        *,
        db_obj: ModelType,
        obj_in: UpdateSchemaType | dict[str, Any]
    ) -> ModelType:
        """Update an existing record."""
        # Normalize input data
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
            
        # Update model attributes
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        self.db.add(db_obj)
        try:
            await self.db.commit()
            await self.db.refresh(db_obj)
            
            self.logger.info(
                "Updated record",
                id=str(getattr(db_obj, "id", None)),
                changes=list(update_data.keys())
            )
            return db_obj
            
        except IntegrityError as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Update operation violated constraints"
            ) from e
            
    async def delete(self, id: Any) -> ModelType | None:
        """Delete a record by primary key."""
        obj = await self.get(id)
        if not obj:
            return None
            
        await self.db.delete(obj)
        await self.db.commit()
        
        self.logger.info(
            "Deleted record",
            id=str(id),
            model=self.model.__name__
        )
        return obj

    async def get_or_404(self, id: Any) -> ModelType:
        """Get record or raise 404."""
        obj = await self.get(id)
        if not obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{self.model.__name__} not found"
            )
        return obj

    async def commit_or_rollback(self) -> bool:
        """Commit current transaction or rollback on error.
        
        Use this to ensure no dangling transactions after service operations.
        
        Returns:
            True if committed successfully, False if rolled back.
        """
        try:
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            self.logger.error(
                "Transaction rolled back",
                error=str(e),
                model=self.model.__name__ if hasattr(self, 'model') else 'unknown'
            )
            return False

    async def flush(self) -> None:
        """Flush pending changes without committing.
        
        Useful for getting auto-generated IDs before commit.
        """
        await self.db.flush()

