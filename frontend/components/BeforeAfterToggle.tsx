'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Scene3DViewer from './Scene3DViewer';

interface BeforeAfterToggleProps {
  beforeModelUrl: string;
  afterModelUrl: string;
  changes?: Array<{
    item: string;
    action: string;
    reason: string;
  }>;
  additions?: Array<{
    item: string;
    reason: string;
    estimated_cost?: number;
  }>;
}

export default function BeforeAfterToggle({
  beforeModelUrl,
  afterModelUrl,
  changes = [],
  additions = [],
}: BeforeAfterToggleProps) {
  const [showAfter, setShowAfter] = useState(false);

  return (
    <div className="w-full h-screen flex flex-col">
      {/* Header */}
      <div className="bg-gray-900 text-white p-6 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Your Room Optimized</h1>
          <p className="text-gray-400 mt-1">
            {showAfter ? 'After' : 'Before'} - Toggle to see the transformation
          </p>
        </div>
        
        {/* Toggle Button */}
        <button
          onClick={() => setShowAfter(!showAfter)}
          className="relative inline-flex h-12 w-48 items-center rounded-full bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          <motion.span
            className="inline-block h-10 w-24 rounded-full bg-blue-500 shadow-lg"
            layout
            transition={{
              type: 'spring',
              stiffness: 700,
              damping: 30,
            }}
            style={{
              x: showAfter ? 88 : 0,
            }}
          />
          <span className="absolute left-6 text-sm font-medium text-white">
            Before
          </span>
          <span className="absolute right-6 text-sm font-medium text-white">
            After
          </span>
        </button>
      </div>

      {/* 3D Viewer */}
      <div className="flex-1 relative bg-gray-100">
        <AnimatePresence mode="wait">
          <motion.div
            key={showAfter ? 'after' : 'before'}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0"
          >
            <Scene3DViewer
              modelUrl={showAfter ? afterModelUrl : beforeModelUrl}
              showGrid={true}
              cameraPosition={[5, 5, 5]}
            />
          </motion.div>
        </AnimatePresence>

        {/* Floating Info Panel */}
        <AnimatePresence>
          {showAfter && (
            <motion.div
              initial={{ x: 300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 300, opacity: 0 }}
              transition={{ type: 'spring', damping: 25 }}
              className="absolute right-4 top-4 w-80 max-h-[80vh] overflow-y-auto bg-white rounded-lg shadow-2xl p-6"
            >
              <h2 className="text-xl font-bold text-gray-900 mb-4">
                ✨ What Changed
              </h2>

              {/* Furniture Changes */}
              {changes.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-sm font-semibold text-gray-700 uppercase mb-3">
                    Rearrangements
                  </h3>
                  <div className="space-y-3">
                    {changes.map((change, idx) => (
                      <div
                        key={idx}
                        className="bg-blue-50 rounded-lg p-3 border-l-4 border-blue-500"
                      >
                        <div className="font-medium text-gray-900 capitalize">
                          {change.item}
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          {change.reason}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* New Additions */}
              {additions.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 uppercase mb-3">
                    Recommended Additions
                  </h3>
                  <div className="space-y-3">
                    {additions.map((addition, idx) => (
                      <div
                        key={idx}
                        className="bg-green-50 rounded-lg p-3 border-l-4 border-green-500"
                      >
                        <div className="flex justify-between items-start">
                          <div className="font-medium text-gray-900 capitalize">
                            {addition.item}
                          </div>
                          {addition.estimated_cost && (
                            <div className="text-sm font-semibold text-green-700">
                              ${addition.estimated_cost}
                            </div>
                          )}
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          {addition.reason}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Total Cost */}
                  {additions.some((a) => a.estimated_cost) && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-gray-900">
                          Total Investment
                        </span>
                        <span className="text-xl font-bold text-green-600">
                          $
                          {additions.reduce(
                            (sum, a) => sum + (a.estimated_cost || 0),
                            0
                          )}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Controls Help */}
      <div className="bg-gray-800 text-gray-400 px-6 py-3 text-sm flex gap-6">
        <span>🖱️ Left click + drag to rotate</span>
        <span>🖱️ Right click + drag to pan</span>
        <span>🖱️ Scroll to zoom</span>
      </div>
    </div>
  );
}


