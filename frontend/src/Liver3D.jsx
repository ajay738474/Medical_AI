import React, { Suspense, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    Center,
    Bounds,
    useGLTF,
    Html,
} from "@react-three/drei";

function LiverModel() {
    const { scene } = useGLTF("/models/liver/liver.glb");

    useEffect(() => {
        scene.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;

                if (child.material) {
                    child.material.roughness = 0.42;
                    child.material.metalness = 0.0;
                }
            }
        });
    }, [scene]);

    return (
        <Center>
            <primitive object={scene} />
        </Center>
    );
}

function Loading() {
    return (
        <Html center>
            <div
                style={{
                    color: "#ffffff",
                    fontSize: "14px",
                    fontWeight: "600",
                    whiteSpace: "nowrap",
                }}
            >
                Loading liver...
            </div>
        </Html>
    );
}

export default function Liver3D() {
    return (
        <div className="liver-3d-container">
            <Canvas
                camera={{
                    position: [0, 0, 4.5],
                    fov: 42,
                    near: 0.1,
                    far: 100,
                }}
                shadows
                dpr={[1, 2]}
            >
                <color attach="background" args={["#101522"]} />

                <ambientLight intensity={1.6} />

                <hemisphereLight
                    intensity={1.8}
                    skyColor="#ffffff"
                    groundColor="#202838"
                />

                <directionalLight
                    position={[5, 6, 5]}
                    intensity={3}
                    castShadow
                />

                <directionalLight
                    position={[-5, 3, 2]}
                    intensity={1.8}
                />

                <pointLight
                    position={[0, 2, 4]}
                    intensity={1.5}
                />

                <Suspense fallback={<Loading />}>
                    <Bounds fit clip observe margin={1.25}>
                        <LiverModel />
                    </Bounds>

                    <Environment preset="studio" />
                </Suspense>

                <OrbitControls
                    enablePan
                    enableZoom
                    enableRotate
                    minDistance={1.5}
                    maxDistance={10}
                    dampingFactor={0.08}
                    enableDamping
                />
            </Canvas>

            <div className="liver-3d-title">
                <span>🫀</span>
                Interactive Human Liver
            </div>

            <div className="liver-3d-help">
                Drag to rotate • Scroll to zoom • Right drag to move
            </div>
        </div>
    );
}

useGLTF.preload("/models/liver/liver.glb");