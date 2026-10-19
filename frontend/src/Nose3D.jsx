import React, { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    useGLTF,
    Center,
} from "@react-three/drei";
import * as THREE from "three";

function NoseModel() {
    const { scene } = useGLTF("/models/nose/nose.glb");

    const model = useMemo(() => {
        const clone = scene.clone(true);

        // Prepare the model's meshes and materials.
        clone.traverse((child) => {
            if (!child.isMesh) return;

            child.castShadow = true;
            child.receiveShadow = true;

            const materials = Array.isArray(child.material)
                ? child.material
                : [child.material];

            materials.forEach((material) => {
                if (!material) return;

                material.side = THREE.DoubleSide;
                material.needsUpdate = true;
            });
        });

        // Calculate the original model bounds.
        clone.updateMatrixWorld(true);

        const bounds = new THREE.Box3().setFromObject(clone);
        const size = bounds.getSize(new THREE.Vector3());
        const maxDimension = Math.max(size.x, size.y, size.z);

        // Normalize model size without changing its geometry.
        if (Number.isFinite(maxDimension) && maxDimension > 0) {
            clone.scale.setScalar(2.5 / maxDimension);
        } else {
            console.error("Nose model has invalid dimensions:", size);
        }

        // Recalculate bounds after scaling.
        clone.updateMatrixWorld(true);

        const scaledBounds = new THREE.Box3().setFromObject(clone);
        const center = scaledBounds.getCenter(new THREE.Vector3());

        // Move the model's center to the origin.
        clone.position.sub(center);
        clone.updateMatrixWorld(true);

        console.log("Nose model original size:", size);
        console.log("Nose model scaled bounds:", scaledBounds);

        return clone;
    }, [scene]);

    return <primitive object={model} />;
}

function LoadingMessage() {
    return (
        <mesh>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color="#60a5fa" />
        </mesh>
    );
}

export default function Nose3D() {
    return (
        <div className="nose-3d-container">
            <Canvas
                camera={{
                    position: [0, 0, 4],
                    fov: 40,
                    near: 0.1,
                    far: 100,
                }}
                dpr={[1, 2]}
                gl={{ antialias: true, alpha: false }}
                onCreated={({ gl }) => {
                    gl.setClearColor("#101522");
                }}
            >
                <color attach="background" args={["#101522"]} />

                <ambientLight intensity={1.8} />

                <directionalLight
                    position={[4, 5, 6]}
                    intensity={2.5}
                />

                <directionalLight
                    position={[-4, 2, 3]}
                    intensity={1.5}
                />

                <directionalLight
                    position={[0, -3, -4]}
                    intensity={1}
                />

                <Suspense fallback={<LoadingMessage />}>
                    <Center>
                        <NoseModel />
                    </Center>

                    <Environment preset="studio" />
                </Suspense>

                <OrbitControls
                    enablePan={false}
                    enableZoom
                    enableRotate
                    minDistance={1}
                    maxDistance={10}
                    target={[0, 0, 0]}
                />
            </Canvas>

            <div className="nose-3d-title">
                👃 Interactive Human Nose
            </div>

            <div className="nose-3d-help">
                Drag to rotate • Scroll to zoom
            </div>
        </div>
    );
}

useGLTF.preload("/models/nose/nose.glb");