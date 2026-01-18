# Room Identity Optimizer - Frontend

Beautiful 3D viewer with before/after toggle for room optimization.

## Features

- ✨ **3D Room Viewer** - Interactive Three.js canvas
- 🔄 **Before/After Toggle** - Smooth animations between layouts
- 📝 **Explanations Panel** - Shows why each change was made
- 💰 **Budget Breakdown** - Displays cost of new furniture
- 🎮 **Intuitive Controls** - Rotate, pan, zoom

## Tech Stack

- **Next.js 14** - React framework
- **Three.js** - 3D rendering
- **@react-three/fiber** - React renderer for Three.js
- **@react-three/drei** - Useful helpers for R3F
- **Framer Motion** - Smooth animations
- **Tailwind CSS** - Styling

## Getting Started

### Install Dependencies

```bash
cd frontend
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Components

### `Scene3DViewer`

Renders a single 3D model with lighting, camera, and controls.

```tsx
<Scene3DViewer
  modelUrl="/models/room.glb"
  showGrid={true}
  cameraPosition={[5, 5, 5]}
/>
```

### `BeforeAfterToggle`

Complete before/after viewer with toggle and explanations.

```tsx
<BeforeAfterToggle
  beforeModelUrl="/models/room_before.glb"
  afterModelUrl="/models/room_after.glb"
  changes={[
    {
      item: 'desk',
      action: 'move',
      reason: 'Moved to reduce glare'
    }
  ]}
  additions={[
    {
      item: 'lamp',
      reason: 'Better task lighting',
      estimated_cost: 40
    }
  ]}
/>
```

## File Structure

```
frontend/
├── app/
│   ├── page.tsx           # Home page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/
│   ├── Scene3DViewer.tsx  # 3D model viewer
│   └── BeforeAfterToggle.tsx  # Before/after component
├── public/
│   └── models/            # 3D model files (.glb)
├── package.json
└── tsconfig.json
```

## Adding 3D Models

Place your `.glb` or `.gltf` files in `public/models/`:

```
public/
└── models/
    ├── room_before.glb
    ├── room_after.glb
    └── furniture/
        ├── bed.glb
        ├── desk.glb
        └── chair.glb
```

Reference them with `/models/filename.glb`

## Integration with Backend

### Fetching Data

```tsx
// In your page component
const [roomData, setRoomData] = useState(null);

useEffect(() => {
  fetch('/api/room/123')
    .then(res => res.json())
    .then(data => setRoomData(data));
}, []);

return (
  <BeforeAfterToggle
    beforeModelUrl={roomData.before_model_url}
    afterModelUrl={roomData.after_model_url}
    changes={roomData.optimization_plan.furniture_changes}
    additions={roomData.optimization_plan.furniture_additions}
  />
);
```

## Customization

### Change Colors

Edit `tailwind.config.ts`:

```ts
theme: {
  extend: {
    colors: {
      primary: '#3B82F6',
      secondary: '#10B981',
    },
  },
},
```

### Adjust Camera

```tsx
<Scene3DViewer
  cameraPosition={[10, 10, 10]}  // Further away
  // or
  cameraPosition={[2, 2, 2]}     // Closer
/>
```

### Change Lighting

Edit `Scene3DViewer.tsx`:

```tsx
<ambientLight intensity={0.8} />  // Brighter ambient
<directionalLight intensity={1.5} />  // Stronger directional
```

## Performance Tips

1. **Optimize 3D Models**
   - Use Draco compression for GLB files
   - Keep poly count < 100k triangles
   - Texture size ≤ 2048x2048

2. **Lazy Load Models**
   - Models load on demand with Suspense
   - Shows loading state automatically

3. **Reduce Draw Calls**
   - Merge meshes when possible
   - Use instancing for repeated objects

## Deployment

### Build for Production

```bash
npm run build
npm start
```

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

## Troubleshooting

### Models Not Loading

- Check file path is correct (`/models/...`)
- Ensure file is in `public/` directory
- Check browser console for errors

### Performance Issues

- Reduce model complexity
- Lower texture resolution
- Disable shadows if needed

### TypeScript Errors

```bash
npm install --save-dev @types/three
```

## Next Steps

- [ ] Add upload interface
- [ ] Add identity selector
- [ ] Add progress indicators
- [ ] Add export/share features
- [ ] Add mobile responsive design

## License

MIT


