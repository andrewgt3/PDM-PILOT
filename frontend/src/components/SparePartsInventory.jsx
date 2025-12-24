import React from 'react';
import { Package, CheckCircle, AlertTriangle, XCircle, Truck, MapPin, Clock } from 'lucide-react';

/**
 * SparePartsInventory Component
 * 
 * Shows spare parts availability for the recommended maintenance.
 * Would integrate with ERP (SAP, Oracle) or inventory system in production.
 */
function SparePartsInventory({ machineId, requiredParts = null }) {
    // Mock parts inventory data
    const partsInventory = requiredParts || [
        {
            partNumber: 'SKF-6206-2RS',
            description: 'Deep Groove Ball Bearing 30x62x16mm',
            quantity: 4,
            location: 'Shelf A3-15',
            status: 'in_stock',
            leadTime: null,
            unitCost: 24.50
        },
        {
            partNumber: 'FANUC-A06B-0243',
            description: 'Servo Motor Drive End Bearing',
            quantity: 1,
            location: 'Shelf B2-08',
            status: 'low_stock',
            reorderPoint: 2,
            leadTime: null,
            unitCost: 185.00
        },
        {
            partNumber: 'LUB-MOBIL-SHC-220',
            description: 'Synthetic Gear Oil 1L',
            quantity: 12,
            location: 'Shelf C1-02',
            status: 'in_stock',
            leadTime: null,
            unitCost: 32.00
        },
        {
            partNumber: 'SEAL-NBR-35x47x7',
            description: 'Shaft Seal NBR 35x47x7mm',
            quantity: 0,
            location: '-',
            status: 'out_of_stock',
            leadTime: '3-5 days',
            unitCost: 8.50,
            onOrder: true,
            expectedDate: '2024-12-28'
        },
        {
            partNumber: 'GREASE-SKF-LGMT2',
            description: 'General Purpose Grease 400g',
            quantity: 8,
            location: 'Shelf C1-01',
            status: 'in_stock',
            leadTime: null,
            unitCost: 15.00
        }
    ];

    const statusConfig = {
        in_stock: {
            icon: CheckCircle,
            color: 'text-emerald-600',
            bg: 'bg-emerald-100',
            label: 'In Stock'
        },
        low_stock: {
            icon: AlertTriangle,
            color: 'text-amber-600',
            bg: 'bg-amber-100',
            label: 'Low Stock'
        },
        out_of_stock: {
            icon: XCircle,
            color: 'text-red-600',
            bg: 'bg-red-100',
            label: 'Out of Stock'
        }
    };

    const inStockCount = partsInventory.filter(p => p.status === 'in_stock').length;
    const outOfStockCount = partsInventory.filter(p => p.status === 'out_of_stock').length;
    const totalValue = partsInventory.reduce((sum, p) => sum + (p.unitCost * Math.max(1, p.quantity)), 0);

    return (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            {/* Header */}
            <div className="px-5 py-4 border-b border-slate-100 bg-slate-50">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Package className="w-5 h-5 text-slate-600" />
                        <h3 className="font-semibold text-slate-900">Spare Parts Inventory</h3>
                    </div>
                    <div className="flex items-center gap-2">
                        {outOfStockCount > 0 && (
                            <span className="flex items-center gap-1 px-2 py-0.5 bg-red-100 text-red-700 text-xs font-medium rounded-full">
                                {outOfStockCount} Pending
                            </span>
                        )}
                        <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-medium rounded-full">
                            {inStockCount}/{partsInventory.length} Available
                        </span>
                    </div>
                </div>
            </div>

            {/* Parts List */}
            <div className="divide-y divide-slate-100 max-h-80 overflow-y-auto">
                {partsInventory.map((part, idx) => {
                    const config = statusConfig[part.status];
                    const StatusIcon = config.icon;
                    return (
                        <div key={idx} className="px-5 py-3 hover:bg-slate-50 transition-colors">
                            <div className="flex items-start justify-between">
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="text-xs font-mono font-medium text-indigo-600">
                                            {part.partNumber}
                                        </span>
                                        <span className={`flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-medium rounded ${config.bg} ${config.color}`}>
                                            <StatusIcon className="w-3 h-3" />
                                            {config.label}
                                        </span>
                                    </div>
                                    <p className="text-sm text-slate-800 truncate">{part.description}</p>
                                    <div className="flex items-center gap-4 mt-1.5 text-xs text-slate-500">
                                        {part.quantity > 0 && (
                                            <span className="flex items-center gap-1">
                                                <MapPin className="w-3 h-3" />
                                                {part.location}
                                            </span>
                                        )}
                                        {part.onOrder && (
                                            <span className="flex items-center gap-1 text-blue-600">
                                                <Truck className="w-3 h-3" />
                                                On Order - ETA {part.expectedDate}
                                            </span>
                                        )}
                                        {part.leadTime && !part.onOrder && (
                                            <span className="flex items-center gap-1 text-amber-600">
                                                <Clock className="w-3 h-3" />
                                                Lead time: {part.leadTime}
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <div className="text-right ml-4">
                                    <div className="text-lg font-bold text-slate-900">
                                        {part.quantity}
                                    </div>
                                    <div className="text-xs text-slate-500">
                                        ${part.unitCost.toFixed(2)} each
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 bg-slate-50 border-t border-slate-100 flex items-center justify-between text-xs">
                <div className="flex items-center gap-4">
                    <span className="text-slate-500">
                        Est. Parts Cost: <span className="font-medium text-slate-700">${totalValue.toFixed(2)}</span>
                    </span>
                </div>
                <button className="text-indigo-600 hover:text-indigo-800 font-medium">View Full Inventory →</button>
            </div>
        </div>
    );
}

export default SparePartsInventory;
