'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, PerspectiveCamera } from '@react-three/drei';
import { Suspense, useState } from 'react';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { useLoader } from '@react-three/fiber';

interface Scene3DViewerProps {
  modelUrl: string;
  showGrid?: boolean;
  cameraPosition?: [number, number, number];
}

function Model({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  return <primitive object={gltf.scene} />;
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="gray" wireframe />
    </mesh>
  );
}

export default function Scene3DViewer({
  modelUrl,
  showGrid = true,
  cameraPosition = [5, 5, 5],
}: Scene3DViewerProps) {
  return (
    <div className="w-full h-full">
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={cameraPosition} />
        
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        
        {/* Environment */}
        <Environment preset="apartment" />
        
        {/* Grid */}
        {showGrid && (
          <gridHelper args={[20, 20, '#888888', '#444444']} />
        )}
        
        {/* 3D Model */}
        <Suspense fallback={<LoadingFallback />}>
          <Model url={modelUrl} />
        </Suspense>
        
        {/* Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={2}
          maxDistance={20}
        />
      </Canvas>
    </div>
  );
}


