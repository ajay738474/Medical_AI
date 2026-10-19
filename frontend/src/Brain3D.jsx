import React, {
    Suspense,
    useEffect,
    useRef,
} from "react";

import { Canvas, useThree } from "@react-three/fiber";

import {
    OrbitControls,
    Environment,
    Html,
    useGLTF,
} from "@react-three/drei";

import * as THREE from "three";


/* ============================================================
   BRAIN MODEL PATH
============================================================ */

const BRAIN_MODEL =
    "/models/brain/brain.glb";


/* ============================================================
   LOADING SCREEN
============================================================ */

function LoadingBrain() {
    return (
        <Html center>

            <div
                style={{
                    color: "white",
                    textAlign: "center",
                    fontFamily:
                        "Arial, sans-serif",

                    background:
                        "rgba(8, 16, 31, 0.92)",

                    padding:
                        "20px 26px",

                    borderRadius:
                        "14px",

                    minWidth:
                        "200px",

                    boxShadow:
                        "0 15px 40px rgba(0,0,0,0.35)",
                }}
            >

                <div
                    style={{
                        fontSize: "32px",
                        marginBottom: "8px",
                    }}
                >
                    🧠
                </div>

                <strong>
                    Loading Brain...
                </strong>

                <div
                    style={{
                        marginTop: "7px",
                        fontSize: "11px",
                        color: "#aebbd0",
                    }}
                >
                    Loading anatomical 3D model
                </div>

            </div>

        </Html>
    );
}


/* ============================================================
   BRAIN MODEL
============================================================ */

function BrainModel() {

    const groupRef =
        useRef(null);

    /*
      Load the NIH brain GLB.
    */

    const { scene } =
        useGLTF(BRAIN_MODEL);


    /* ==========================================================
       MODEL SETUP
    ========================================================== */

    useEffect(() => {

        if (!scene) {
            return;
        }


        /* --------------------------------------------------------
           MATERIAL SETTINGS
        -------------------------------------------------------- */

        scene.traverse(
            (object) => {

                if (!object.isMesh) {
                    return;
                }


                object.castShadow = true;

                object.receiveShadow = true;


                const materials =
                    Array.isArray(
                        object.material
                    )
                        ? object.material
                        : [object.material];


                materials.forEach(
                    (material) => {

                        if (!material) {
                            return;
                        }


                        material.side =
                            THREE.DoubleSide;


                        material.roughness =
                            0.72;


                        material.metalness =
                            0.02;

                    }
                );

            }
        );


        /* ========================================================
           AUTOMATIC CENTER + SCALE
        ======================================================== */

        /*
          Reset transformation first.
        */

        scene.scale.set(
            1,
            1,
            1
        );

        scene.position.set(
            0,
            0,
            0
        );


        /*
          Calculate the real dimensions
          of the downloaded GLB.
        */

        const box =
            new THREE.Box3()
                .setFromObject(scene);


        const size =
            new THREE.Vector3();


        const center =
            new THREE.Vector3();


        box.getSize(size);

        box.getCenter(center);


        const maxDimension =
            Math.max(
                size.x,
                size.y,
                size.z
            );


        console.log(
            "Brain original size:",
            {
                x: size.x,
                y: size.y,
                z: size.z,
            }
        );


        /* ========================================================
           SCALE MODEL
        ======================================================== */

        if (
            maxDimension > 0
        ) {

            /*
              Target size inside
              the 3D viewer.
      
              2.3 gives a comfortable
              brain size.
            */

            const targetSize =
                2.3;


            const scale =
                targetSize /
                maxDimension;


            scene.scale.setScalar(
                scale
            );


            /* ------------------------------------------------------
               CENTER MODEL
            ------------------------------------------------------ */

            scene.position.set(
                -center.x * scale,
                -center.y * scale,
                -center.z * scale
            );


            console.log(
                "Brain scale:",
                scale
            );

        }


    }, [scene]);


    return (

        <group
            ref={groupRef}
        >

            <primitive
                object={scene}
            />

        </group>

    );
}


/* ============================================================
   BRAIN SCENE
============================================================ */

function BrainScene() {

    const { camera } =
        useThree();


    /* ==========================================================
       CAMERA
    ========================================================== */

    useEffect(() => {

        camera.position.set(
            0,
            0.2,
            6
        );


        camera.lookAt(
            0,
            0,
            0
        );

    }, [camera]);


    return (
        <>

            {/* ======================================================
          LIGHTING
      ====================================================== */}

            <ambientLight
                intensity={1.8}
            />


            <directionalLight
                position={[
                    5,
                    6,
                    7,
                ]}
                intensity={2.5}
                castShadow
            />


            <directionalLight
                position={[
                    -5,
                    2,
                    4,
                ]}
                intensity={1.4}
            />


            <pointLight
                position={[
                    0,
                    2,
                    6,
                ]}
                intensity={1.5}
            />


            {/* ======================================================
          BRAIN
      ====================================================== */}

            <Suspense
                fallback={
                    <LoadingBrain />
                }
            >

                <BrainModel />

            </Suspense>


            {/* ======================================================
          INTERACTION
      ====================================================== */}

            <OrbitControls

                enableRotate

                enableZoom

                enablePan


                minDistance={2}

                maxDistance={10}


                enableDamping

                dampingFactor={0.08}


                rotateSpeed={0.7}

                zoomSpeed={0.8}

                panSpeed={0.7}


                target={[
                    0,
                    0,
                    0,
                ]}

            />


            {/* ======================================================
          STUDIO ENVIRONMENT
      ====================================================== */}

            <Environment
                preset="studio"
            />

        </>
    );
}


/* ============================================================
   ERROR BOUNDARY
============================================================ */

class BrainErrorBoundary
    extends React.Component {

    constructor(props) {

        super(props);

        this.state = {
            hasError: false,
            errorMessage: "",
        };

    }


    static getDerivedStateFromError(
        error
    ) {

        return {
            hasError: true,

            errorMessage:
                error?.message ||
                "Unknown brain model error",
        };

    }


    componentDidCatch(
        error,
        errorInfo
    ) {

        console.error(
            "BRAIN 3D ERROR:",
            error
        );

        console.error(
            "BRAIN ERROR INFO:",
            errorInfo
        );

    }


    render() {

        if (
            this.state.hasError
        ) {

            return (

                <div
                    style={{
                        width: "100%",
                        height: "100%",
                        minHeight: "300px",

                        display: "flex",

                        alignItems:
                            "center",

                        justifyContent:
                            "center",

                        textAlign:
                            "center",

                        color: "white",

                        background:
                            "radial-gradient(circle at center, #253b61, #0b1426)",

                        padding: "30px",

                        boxSizing:
                            "border-box",
                    }}
                >

                    <div>

                        <div
                            style={{
                                fontSize: "48px",
                                marginBottom: "12px",
                            }}
                        >
                            🧠
                        </div>


                        <h3
                            style={{
                                margin:
                                    "0 0 8px",
                            }}
                        >
                            Brain model could not load
                        </h3>


                        <p
                            style={{
                                margin: "0 0 12px",
                                color: "#aebbd0",
                                fontSize: "12px",
                            }}
                        >
                            Please check the brain
                            GLB file.
                        </p>


                        <div
                            style={{
                                maxWidth: "280px",
                                margin: "0 auto",

                                padding:
                                    "8px 10px",

                                borderRadius:
                                    "8px",

                                background:
                                    "rgba(255,255,255,0.06)",

                                color:
                                    "#8fa3bf",

                                fontSize: "9px",

                                wordBreak:
                                    "break-word",
                            }}
                        >
                            {this.state.errorMessage}
                        </div>

                    </div>

                </div>

            );

        }


        return this.props.children;

    }

}


/* ============================================================
   MAIN BRAIN 3D COMPONENT
============================================================ */

export default function Brain3D() {

    return (

        <div
            className="brain-3d-container"
            style={{
                position: "relative",
                width: "100%",
                height: "100%",
            }}
        >

            {/* ======================================================
          THREE.JS CANVAS
      ====================================================== */}

            <BrainErrorBoundary>

                <Canvas

                    camera={{
                        position: [
                            0,
                            0.2,
                            6,
                        ],

                        fov: 45,

                        near: 0.01,

                        far: 100,
                    }}


                    dpr={[
                        1,
                        2,
                    ]}


                    gl={{
                        antialias: true,

                        alpha: true,

                        powerPreference:
                            "high-performance",
                    }}


                    shadows

                >

                    <BrainScene />

                </Canvas>

            </BrainErrorBoundary>


            {/* ======================================================
          INFORMATION OVERLAY
      ====================================================== */}

            <div
                className="brain-3d-controls"
                style={{
                    position: "absolute",

                    left: "16px",

                    right: "16px",

                    bottom: "16px",

                    pointerEvents: "none",

                    padding:
                        "12px 15px",

                    borderRadius:
                        "12px",

                    background:
                        "rgba(7, 15, 29, 0.82)",

                    backdropFilter:
                        "blur(10px)",

                    border:
                        "1px solid rgba(255,255,255,0.08)",

                    boxShadow:
                        "0 8px 25px rgba(0,0,0,0.25)",
                }}
            >

                <div
                    className="brain-3d-title"
                    style={{
                        color: "#ffffff",

                        fontSize: "13px",

                        fontWeight: 750,
                    }}
                >
                    🧠 Interactive Human Brain
                </div>


                <div
                    className="brain-3d-help"
                    style={{
                        marginTop: "5px",

                        color: "#a9b7ca",

                        fontSize: "10px",

                        lineHeight: 1.5,
                    }}
                >
                    Drag to rotate • Scroll to zoom •
                    Right-click to pan
                </div>


                <div
                    className="brain-3d-source"
                    style={{
                        marginTop: "4px",

                        color: "#7187a5",

                        fontSize: "9px",
                    }}
                >
                    Anatomical reference model • NIH 3D
                </div>

            </div>

        </div>

    );
}


/* ============================================================
   PRELOAD
============================================================ */

useGLTF.preload(
    BRAIN_MODEL
);