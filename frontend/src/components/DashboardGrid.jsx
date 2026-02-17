import React, { useState, useEffect, useRef } from 'react';
import {
    DndContext,
    useSensor,
    useSensors,
    PointerSensor,
    useDraggable,
} from '@dnd-kit/core';
import { Box } from '@mui/material';
import { GripVertical, Scaling } from 'lucide-react';

/**
 * FreeDraggableWidget
 * A clear wrapper for free-form dragging and resizing.
 */
function FreeDraggableWidget({ id, left, top, width, height, children, dragHandle = true, onResize }) {
    const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
        id: id,
    });

    const [isResizing, setIsResizing] = useState(false);
    // Local state for smooth resizing without waiting for parent round-trip
    const [localSize, setLocalSize] = useState(null);

    // Sync local size with props when not resizing.
    // We intentionally omit isResizing from the dependency array so that we DO NOT
    // reset to stale props immediately when resizing ends. We only want to update
    // when the parent actually confirms the new size with new props.
    useEffect(() => {
        if (!isResizing) {
            setLocalSize({ width: width || 400, height: height || 'auto' });
        }
    }, [width, height]); // Removed isResizing to prevent stale snap-back

    // Calculate current display dimensions
    // Always use localSize if it exists (it acts as optimistic UI), otherwise fallback to props
    const currentWidth = localSize ? localSize.width : (width || 400);
    const currentHeight = localSize ? localSize.height : (height || 'auto');

    const style = {
        position: 'absolute',
        top: top,
        left: left,
        zIndex: isDragging ? 1000 : 1,
        // Translate mimics the drag movement
        transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
        width: currentWidth,
        height: currentHeight,
        touchAction: 'none', // Critical for drag interactions
        cursor: isDragging ? 'grabbing' : 'auto',
        boxShadow: isDragging ? '0px 10px 20px rgba(0,0,0,0.2)' : 'none',
        // Important: disable transition during resize for 1:1 feel
        transition: isDragging || isResizing ? 'none' : 'box-shadow 0.3s, width 0.1s, height 0.1s',
    };

    // Resize Handler using Pointer Events with Capture
    const handleResizeStart = (e) => {
        // Stop creating dragging conflicts
        e.stopPropagation();
        e.preventDefault();

        const target = e.currentTarget;
        const startX = e.clientX;
        const startY = e.clientY;
        const startWidth = width || 400;
        const startHeight = height || 400;

        setIsResizing(true);
        setLocalSize({ width: startWidth, height: startHeight });

        // Capture pointer to ensure smooth dragging even if mouse leaves the element
        try {
            target.setPointerCapture(e.pointerId);
        } catch (err) {
            console.error('Failed to capture pointer', err);
        }

        let animationFrameId;

        const handlePointerMove = (moveEvent) => {
            moveEvent.preventDefault();
            moveEvent.stopPropagation();

            // Use requestAnimationFrame to prevent event flooding
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            animationFrameId = requestAnimationFrame(() => {
                // Calculate delta
                const deltaX = moveEvent.clientX - startX;
                const deltaY = moveEvent.clientY - startY;

                const newWidth = Math.max(300, startWidth + deltaX);
                const newHeight = Math.max(200, startHeight + deltaY);

                // Update local state immediately for UI responsiveness
                setLocalSize({ width: newWidth, height: newHeight });

                // Propagate to parent (this allows chart to reflow)
                onResize(id, newWidth, newHeight);
            });
        };

        const handlePointerUp = (upEvent) => {
            setIsResizing(false);
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            try {
                if (target.hasPointerCapture && target.hasPointerCapture(upEvent.pointerId)) {
                    target.releasePointerCapture(upEvent.pointerId);
                }
            } catch (err) {
                // ignore
            }

            target.removeEventListener('pointermove', handlePointerMove);
            target.removeEventListener('pointerup', handlePointerUp);

            // FORCE FINAL UPDATE on release to prevent snap-back
            // This ensures that even if a RAF frame was missed or the parent state is stale,
            // we enforce the final position one last time.
            const deltaX = upEvent.clientX - startX;
            const deltaY = upEvent.clientY - startY;
            const finalWidth = Math.max(300, startWidth + deltaX);
            const finalHeight = Math.max(200, startHeight + deltaY);

            setLocalSize({ width: finalWidth, height: finalHeight }); // Keep local state correct until props update
            onResize(id, finalWidth, finalHeight);
        };

        // Attach listeners to target instead of window when using capture
        target.addEventListener('pointermove', handlePointerMove);
        target.addEventListener('pointerup', handlePointerUp);
    };

    return (
        <Box ref={setNodeRef} style={style}>
            <Box sx={{ position: 'relative', height: '100%', border: isResizing || isDragging ? '1px dashed #3b82f6' : 'none' }}>
                {dragHandle && (
                    <Box
                        {...listeners}
                        {...attributes}
                        sx={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            zIndex: 1200,
                            cursor: 'grab',
                            opacity: 0.6,
                            '&:hover': { opacity: 1, bgcolor: 'action.hover' },
                            color: 'text.secondary',
                            borderRadius: 1,
                            p: 0.5,
                            transition: 'all 0.2s',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}
                    >
                        <GripVertical size={16} />
                    </Box>
                )}

                {children}

                {/* Resize Handle */}
                <Box
                    onPointerDown={handleResizeStart}
                    sx={{
                        position: 'absolute',
                        bottom: 4,
                        right: 4,
                        width: 24,
                        height: 24,
                        zIndex: 1300,
                        cursor: 'nwse-resize',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'text.secondary',
                        opacity: 0.5,
                        '&:hover': { opacity: 1, color: 'primary.main' },
                        // Increase hit area for easier grabbing
                        p: 1,
                        m: -1
                    }}
                >
                    <Scaling size={16} style={{ transform: 'rotate(90deg)' }} />
                </Box>
            </Box>
        </Box>
    );
}

/**
 * DashboardGrid (Refactored to FreeCanvas)
 * 
 * @param {Object} items - Dictionary of items { id: { content, x, y, width, height } }
 * @param {Function} onUpdate - Callback (id, updates) => void. updates = { x, y } or { width, height }
 */
export function DashboardGrid({ items, onUpdate, children }) {
    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                distance: 5,
            },
        })
    );

    const handleDragEnd = (event) => {
        const { active, delta } = event;
        const id = active.id;

        const currentItem = items[id];
        if (!currentItem) return;

        const newX = (currentItem.x || 0) + delta.x;
        const newY = (currentItem.y || 0) + delta.y;

        onUpdate(id, { x: newX, y: newY });
    };

    const handleResize = (id, width, height) => {
        onUpdate(id, { width, height });
    };

    return (
        <DndContext
            sensors={sensors}
            onDragEnd={handleDragEnd}
        >
            <Box sx={{
                position: 'relative',
                width: '100%',
                minHeight: '100vh', // Full screen feel
                bgcolor: 'background.default',
                // Remove the "small" feel - make it a backdrop
                backgroundImage: 'radial-gradient(#e5e7eb 1px, transparent 1px)', // Subtle dot grid for whiteboard feel
                backgroundSize: '20px 20px',
                borderRadius: 2,
                overflow: 'visible' // Allow dragging outside slightly if needed
            }}>
                {children}

                {Object.values(items).map((item) => (
                    <FreeDraggableWidget
                        key={item.id}
                        id={item.id}
                        left={item.x}
                        top={item.y}
                        width={item.width}
                        height={item.height}
                        onResize={handleResize}
                    >
                        {item.content}
                    </FreeDraggableWidget>
                ))}
            </Box>
        </DndContext>
    );
}
