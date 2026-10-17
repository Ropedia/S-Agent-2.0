window.SAGENT_TRAJ = {
 "id": "MMSI_375",
 "bench": "MMSI-Bench",
 "category": "Motion · object",
 "backbone": "Gemini-3.8-Flash",
 "harness": "OpenHands",
 "question": "The images are taken in a continuous first-person perspective. Has the lawn mower been rotated? If it has, in which direction was it rotated?",
 "options": [
  {
   "key": "A",
   "text": "Flipped upside down"
  },
  {
   "key": "B",
   "text": "Rotated counterclockwise"
  },
  {
   "key": "C",
   "text": "Rotated clockwise"
  },
  {
   "key": "D",
   "text": "Not rotated"
  }
 ],
 "gt": "C",
 "pred": "C",
 "correct": true,
 "inputs": [
  {
   "src": "img/MMSI_375/frame_0.jpg",
   "cap": "frame_0"
  },
  {
   "src": "img/MMSI_375/frame_1.jpg",
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
   "note": "Fuses every frame into one metric, gravity-aligned scene; each camera becomes a named instance."
  },
  {
   "kind": "tool",
   "cmd": "segment",
   "line": "perception.py segment --prompt \"lawn mower\"",
   "reply": "…\n\"name\": \"lawn mower\",\n…",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_375/segment_overlay_1354e3fef92a.jpg",
     "cap": "segment overlay · f0"
    },
    {
     "src": "img/MMSI_375/segment_overlay_7c04b98765c7.jpg",
     "cap": "segment overlay · f1"
    }
   ],
   "note": "Segments the lawn mower in both photos."
  },
  {
   "kind": "view",
   "cmd": "view",
   "line": "view /app/frame_0.png\nview /app/frame_1.png",
   "reply": "",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_375/frame_0.jpg",
     "cap": "frame_0"
    },
    {
     "src": "img/MMSI_375/frame_1.jpg",
     "cap": "frame_1"
    }
   ],
   "note": "The agent opens both photos itself."
  },
  {
   "kind": "readout",
   "cmd": "bev",
   "line": "perception.py bev --frame 0",
   "reply": "…\n\"bev\": \"/perception/renders/bev_0d391c69cf57.png\",\n…\n\"label\": \"lawn mower\",\n…",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_375/bev_0d391c69cf57.jpg",
     "cap": "top-down map · frame 0"
    }
   ],
   "note": "Renders a top-down map of the fused scene with every registered instance."
  },
  {
   "kind": "readout",
   "cmd": "motion",
   "line": "perception.py motion --of camera:f0 --of camera:f1",
   "reply": "…\n\"conclusion\": \"from camera:f0 to camera:f1 the camera moved 0.12 m FORWARD, 0.54 m to its LEFT; turned 14 deg CLOCKWISE (to its right); tilted its view 5 deg DOWN; its sideways lean changed by +20 deg (account for this before comparing angles between the tw…\n…",
   "conclusion": "from camera:f0 to camera:f1 the camera moved 0.12 m FORWARD, 0.54 m to its LEFT; turned 14 deg CLOCKWISE (to its right); tilted its view 5 deg DOWN; its sideways lean changed by +20 deg (account for this before comparing angles between the two photos)",
   "images": [],
   "note": "Reads the camera's own motion first: it turned 14° clockwise and its sideways lean changed by 20°."
  },
  {
   "kind": "readout",
   "cmd": "motion",
   "line": "perception.py motion --of f0:0 --of f1:0",
   "reply": "…\n\"conclusion\": \"f0:0 lawn mower -> f1:0 lawn mower rotated 28 deg CLOCKWISE seen from above; tilted 12 deg off vertical; seen from camera f0, rotated 27 deg CLOCKWISE in the image plane\",\n…",
   "conclusion": "f0:0 lawn mower -> f1:0 lawn mower rotated 28 deg CLOCKWISE seen from above; tilted 12 deg off vertical; seen from camera f0, rotated 27 deg CLOCKWISE in the image plane",
   "images": [],
   "note": "Reads the mower's motion between frames (f0:0 → f1:0): rotated 28° clockwise seen from above, 27° clockwise in the image plane."
  },
  {
   "kind": "readout",
   "cmd": "bev",
   "line": "perception.py bev --frame 0 --erase f1:0 && perception.py bev --frame 0 --erase f0:0",
   "reply": "…\n\"bev\": \"/perception/renders/bev_6193f369bccb.png\",\n…\n\"label\": \"lawn mower\",\n…\n\"bev\": \"/perception/renders/bev_2b28d21d0600.png\",\n…",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_375/bev_6193f369bccb.jpg",
     "cap": "top-down map · frame 0 · f1:0 erased"
    },
    {
     "src": "img/MMSI_375/bev_2b28d21d0600.jpg",
     "cap": "top-down map · frame 0 · f0:0 erased"
    }
   ],
   "note": "Renders the two split maps: one ground frame, each showing the mower at one moment, so the camera motion is factored out."
  },
  {
   "kind": "view",
   "cmd": "view",
   "line": "view /perception/renders/bev_6193f369bccb.png\nview /perception/renders/bev_2b28d21d0600.png",
   "reply": "",
   "conclusion": "",
   "images": [
    {
     "src": "img/MMSI_375/bev_6193f369bccb.jpg",
     "cap": "top-down map · frame 0 · f1:0 erased"
    },
    {
     "src": "img/MMSI_375/bev_2b28d21d0600.jpg",
     "cap": "top-down map · frame 0 · f0:0 erased"
    }
   ],
   "note": "Compares the two split maps."
  },
  {
   "kind": "answer",
   "cmd": "answer",
   "line": "echo -n \"C\" > /app/answer.txt",
   "reply": "",
   "conclusion": "",
   "images": [],
   "note": "Writes C (rotated clockwise)."
  }
 ],
 "final": {
  "answer": "C",
  "text": "Based on 3D reconstruction and motion analysis using the perception API:\n- The lawn mower was segmented across `frame_0.png` and `frame_1.png`.\n- Evaluating object motion between the two frames shows that the lawn mower rotated approximately 28° clockwise as seen from above (and 27° clockwise in the image plane).\n- The split bird's-eye-view maps verify that the heading of the lawn mower turned clockwise to the …"
 },
 "stats": {
  "steps": 13,
  "perception_calls": 7,
  "image_views": 4
 }
};
