import React from 'react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Box } from '@mui/material';
import { GripVertical } from 'lucide-react';

/**
 * DraggableWidget
 * Wrapper component that makes its children draggable within a SortableContext.
 * Use this for the list items (logic + UI).
 */
export function DraggableWidget({ id, children, dragHandle = true }) {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging
    } = useSortable({ id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
        zIndex: isDragging ? 1000 : 'auto',
        position: 'relative',
        opacity: isDragging ? 0.3 : 1, // Dim the original placeholder
        height: '100%'
    };

    return (
        <Box ref={setNodeRef} style={style} sx={{ height: '100%' }}>
            <WidgetFrame
                dragHandle={dragHandle}
                isOverlay={false}
                dragListeners={listeners}
                dragAttributes={attributes}
            >
                {children}
            </WidgetFrame>
        </Box>
    );
}

/**
 * WidgetFrame
 * Pure UI component for the widget wrapper. 
 * Used by DraggableWidget (with listeners) and DragOverlay (without listeners/visual only).
 */
export function WidgetFrame({ children, dragHandle, isOverlay, dragListeners, dragAttributes }) {
    // If overlay, apply specific overlay styles
    const overlayProps = isOverlay ? {
        transform: 'scale(1.02)',
        boxShadow: 24,
        borderRadius: 2,
        bgcolor: 'background.paper',
        height: '100%'
    } : {
        position: 'relative',
        height: '100%'
    };

    return (
        <Box sx={overlayProps}>
            <Box sx={{ position: 'relative', height: '100%', width: '100%' }}>
                {dragHandle && (
                    <Box
                        {...dragListeners}
                        {...dragAttributes}
                        sx={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            zIndex: 1200,
                            cursor: isOverlay ? 'grabbing' : 'grab',
                            opacity: isOverlay ? 1 : 0.4,
                            '&:hover': { opacity: 1, bgcolor: 'action.hover' },
                            color: 'text.secondary',
                            borderRadius: 1,
                            p: 0.5,
                            transition: 'all 0.2s',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            touchAction: 'none' // Prevent scroll interference
                        }}
                    >
                        <GripVertical size={16} />
                    </Box>
                )}
                {children}
            </Box>
        </Box>
    );
}
