'use client';

import BeforeAfterToggle from '../components/BeforeAfterToggle';

export default function Home() {
  // Example data - in production, this would come from your API
  const exampleChanges = [
    {
      item: 'desk',
      action: 'move',
      reason: 'Moved away from east window to reduce morning glare and improve focus',
    },
    {
      item: 'bed',
      action: 'move',
      reason: 'Repositioned away from window to avoid morning sun disruption for night owl schedule',
    },
    {
      item: 'chair',
      action: 'rotate',
      reason: 'Rotated to face wall for better concentration during study sessions',
    },
  ];

  const exampleAdditions = [
    {
      item: 'blackout curtain',
      reason: 'Essential for night owl lifestyle - blocks morning light for better sleep',
      estimated_cost: 60,
    },
    {
      item: 'desk lamp',
      reason: 'Provides focused task lighting for night studying without harsh overhead lights',
      estimated_cost: 40,
    },
    {
      item: 'bookshelf',
      reason: 'Organizes study materials and keeps desk clear for focused work',
      estimated_cost: 150,
    },
  ];

  return (
    <main className="w-full h-screen">
      <BeforeAfterToggle
        beforeModelUrl="/models/room_before.glb"
        afterModelUrl="/models/room_after.glb"
        changes={exampleChanges}
        additions={exampleAdditions}
      />
    </main>
  );
}


