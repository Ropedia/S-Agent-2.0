window.SAGENT_TRAJ = {
 "id": "MMSI_363",
 "bench": "MMSI-Bench",
 "category": "Motion · camera",
 "backbone": "Gemini-3.8-Flash",
 "harness": "OpenHands",
 "question": "The images are taken continuously from a first-person perspective. In which direction are you moving (with the front in image 1 as the forward direction)?",
 "options": [
  {
   "key": "A",
   "text": "Front right"
  },
  {
   "key": "B",
   "text": "Right"
  },
  {
   "key": "C",
   "text": "Front left"
  },
  {
   "key": "D",
   "text": "Forward"
  }
 ],
 "gt": "C",
 "pred": "C",
 "correct": true,
 "inputs": [
  {
   "src": "img/MMSI_363/frame_0.jpg",
   "cap": "frame_0"
  },
  {
   "src": "img/MMSI_363/frame_1.jpg",
   "cap": "frame_1"
  }
 ],
 "steps": [
  {
   "kind": "tool",
   "cmd": "recon",
   "line": "perception.py recon",
   "reply": "…\n\"num_frames\": 2,\n…\n\"name\": \"camera:f0\",\n…\n\"name\": \"camera:f1\",\n…\n\"name\": \"scene\",\n…",
   "conclusion": "",
   "images": [],
   "note": "Fuses the two photos into one metric scene; both cameras become instances."
  },
  {
   "kind": "readout",
   "cmd": "motion",
   "line": "perception.py motion --of camera:f0 --of camera:f1",
   "reply": "…\n\"conclusion\": \"from camera:f0 to camera:f1 the camera moved 0.73 m FORWARD, 0.31 m to its LEFT, 0.67 m UP; turned 8 deg CLOCKWISE (to its right); tilted its view 20 deg DOWN; its sideways lean changed by -5 deg (account for this before comparing angles betw…\n…",
   "conclusion": "from camera:f0 to camera:f1 the camera moved 0.73 m FORWARD, 0.31 m to its LEFT, 0.67 m UP; turned 8 deg CLOCKWISE (to its right); tilted its view 20 deg DOWN; its sideways lean changed by -5 deg (account for this before comparing angles between the two photos)",
   "images": [],
   "note": "Reads the camera's motion between the shots: 0.73 m forward and 0.31 m to the left, turning 8° clockwise."
  },
  {
   "kind": "readout",
   "cmd": "bev",
   "line": "perception.py bev --frame 0",
   "reply": "…\n\"code\": \"bad_request\",\n\"message\": \"no saved masks found for any fused frame; run /segment first\",\n…",
   "conclusion": "",
   "images": [],
   "note": "The map needs at least one segmented object, so this call returns an error."
  },
  {
   "kind": "view",
   "cmd": "view",
   "line": "view /app/frame_0.png\nview /app/frame_1.png",
   "reply": "",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_363/frame_0.jpg",
     "cap": "frame_0"
    },
    {
     "src": "img/MMSI_363/frame_1.jpg",
     "cap": "frame_1"
    }
   ],
   "note": "The agent opens both photos itself."
  },
  {
   "kind": "tool",
   "cmd": "segment",
   "line": "perception.py segment --prompt \"car\" --prompt \"tire\"",
   "reply": "…\n\"name\": \"tire\",\n…\n\"name\": \"car\",\n…",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_363/segment_overlay_f81b731e7ca2.jpg",
     "cap": "segment overlay · f0"
    },
    {
     "src": "img/MMSI_363/segment_overlay_dd21f6fb5983.jpg",
     "cap": "segment overlay · f1"
    }
   ],
   "note": "Segments the car and a tire so the map has landmarks."
  },
  {
   "kind": "readout",
   "cmd": "bev",
   "line": "perception.py bev --frame 0",
   "reply": "…\n\"bev\": \"/perception/renders/bev_a1e6ba330015.png\",\n…\n\"label\": \"car\",\n…\n\"label\": \"tire\",\n…",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_363/bev_a1e6ba330015.jpg",
     "cap": "top-down map · frame 0"
    }
   ],
   "note": "Renders a top-down map of the fused scene with every registered instance."
  },
  {
   "kind": "view",
   "cmd": "view",
   "line": "view /perception/renders/bev_a1e6ba330015.png",
   "reply": "",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_363/bev_a1e6ba330015.jpg",
     "cap": "top-down map · frame 0"
    }
   ],
   "note": "Reads the map."
  },
  {
   "kind": "readout",
   "cmd": "side",
   "line": "perception.py side --view 1 --of camera:f0",
   "reply": "…\n\"direction\": \"rear-right\",\n\"conclusion\": \"camera:f0 is 0.68 m BEHIND camera:f1, 0.41 m to its RIGHT and 0.67 m BELOW it — overall to its REAR-RIGHT\"\n…",
   "conclusion": "camera:f0 is 0.68 m BEHIND camera:f1, 0.41 m to its RIGHT and 0.67 m BELOW it — overall to its REAR-RIGHT",
   "images": [],
   "note": "Cross-check from the second camera: the first camera position lies behind and to the right, so the motion was forward and to the left."
  },
  {
   "kind": "answer",
   "cmd": "answer",
   "line": "echo -n \"C\" > /app/answer.txt",
   "reply": "",
   "conclusion": "",
   "images": [],
   "note": "Writes C (front left)."
  }
 ],
 "final": {
  "answer": "C",
  "text": "Based on the 3D scene reconstruction and camera motion analysis, moving from the first image (`frame_0.png`) to the second image (`frame_1.png`) involves translation forward and to the left (0.73 m forward, 0.31 m left in the first camera's frame). Therefore, the direction of movement relative to the front direction in image 1 is Front left (Option C)."
 },
 "stats": {
  "steps": 12,
  "perception_calls": 6,
  "image_views": 3
 }
};
