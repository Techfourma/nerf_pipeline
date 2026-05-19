import controlP5.*;
import java.io.File;

// ================= GLOBAL VARIABLES =================
PImage[] frames;
JSONObject metrics;
int currentFrame = 0;
int totalFrames = 0;
boolean isPlaying = true;
float playbackSpeed = 1.0;
int lastFrameTime = 0;

float rotationX = 0;
float rotationY = 0;
float zoom = 1.0;

float currentPSNR = 0;
float currentSSIM = 0;
float avgPSNR = 0;
float avgSSIM = 0;

ControlP5 cp5;

// ================= SETUP =================
void setup() {
  size(1200, 800, P3D);
  frameRate(30);
  smooth();
  
  cp5 = new ControlP5(this);
  
  loadMetrics();
  loadFrames();
  setupUI();
  
  println("✅ NeRF Visualizer Ready!");
  println("📊 Total frames: " + totalFrames);
  println("📈 Avg PSNR: " + avgPSNR + " dB");
}

void draw() {
  background(30);
  displayRender();
  displayMetricsOverlay();
  displayInfoPanel();
  
  if (isPlaying && millis() - lastFrameTime > 1000/playbackSpeed) {
    currentFrame = (currentFrame + 1) % totalFrames;
    lastFrameTime = millis();
  }
}

// ================= LOAD METRICS (FIXED!) =================
void loadMetrics() {
  println("🔍 Loading metrics...");
  String path = "/home/jundi-lesmana/nerf_pipeline/results/metrics.json";
  
  try {
    // Cek file dulu
    File f = new File(path);
    if (!f.exists()) {
      println("❌ File not found: " + path);
      useDefaults();
      return;
    }
    
    // FIX: loadJSONObject membaca seluruh file, bukan hanya baris 1
    metrics = loadJSONObject(path);
    
    if (metrics != null) {
      JSONObject avg = metrics.getJSONObject("average_metrics");
      avgPSNR = (float) avg.getDouble("psnr");
      avgSSIM = (float) avg.getDouble("ssim");
      
      JSONArray arr = metrics.getJSONArray("per_frame_metrics");
      totalFrames = arr.size();
      
      // Test ambil nilai frame 0 untuk debug
      float testP = (float) arr.getJSONObject(0).getDouble("psnr");
      println("✅ SUCCESS! Loaded " + totalFrames + " frames.");
      println("   Frame 0 PSNR: " + testP);
    } else {
      println(" Failed to parse JSON.");
      useDefaults();
    }
  } catch (Exception e) {
    println("❌ Error: " + e.getMessage());
    useDefaults();
  }
}

void useDefaults() {
  println("⚠️ Using default metrics");
  totalFrames = 40;
  avgPSNR = 22.0;
  avgSSIM = 0.82;
  metrics = null;
}

// ================= LOAD FRAMES =================
void loadFrames() {
  println("🔍 Loading frames...");
  String folder = "/home/jundi-lesmana/nerf_pipeline/results/frames/";
  
  File f = new File(folder);
  if (!f.exists() || !f.isDirectory()) {
    println(" Frames folder not found: " + folder);
    return;
  }
  
  frames = new PImage[totalFrames];
  int count = 0;
  
  for (int i = 0; i < totalFrames; i++) {
    String fname = String.format("frame_%04d.png", i);
    PImage img = loadImage(folder + fname);
    if (img != null) {
      frames[i] = img;
      count++;
    }
  }
  println("✅ Loaded " + count + "/" + totalFrames + " frames");
}

// ================= UI SETUP =================
void setupUI() {
  cp5.addSlider("frameSlider")
    .setPosition(20, 750)
    .setSize(800, 20)
    .setRange(0, totalFrames - 1)
    .setValue(0)
    .setLabel("Frame");
  
  cp5.addButton("playButton")
    .setPosition(850, 745)
    .setSize(80, 30)
    .setLabel(isPlaying ? "Pause" : "Play");
  
  cp5.addSlider("speedSlider")
    .setPosition(950, 750)
    .setSize(150, 20)
    .setRange(0.1, 3.0)
    .setValue(1.0)
    .setLabel("Speed");
  
  cp5.addSlider("zoomSlider")
    .setPosition(20, 720)
    .setSize(200, 15)
    .setRange(0.1, 3.0)
    .setValue(1.0)
    .setLabel("Zoom");
}

// ================= DISPLAY RENDER =================
void displayRender() {
  pushMatrix();
  translate(width/2, height/2, 0);
  rotateX(rotationX);
  rotateY(rotationY);
  scale(zoom);
  
  if (frames != null && currentFrame < frames.length && frames[currentFrame] != null) {
    PImage img = frames[currentFrame];
    image(img, -img.width/2, -img.height/2);
    
    noFill();
    stroke(255, 100);
    strokeWeight(2);
    rect(-img.width/2, -img.height/2, img.width, img.height);
  }
  popMatrix();
}

// ================= METRICS OVERLAY =================
void displayMetricsOverlay() {
  // Ambil metrics per-frame
  try {
    if (metrics != null) {
      JSONArray arr = metrics.getJSONArray("per_frame_metrics");
      if (currentFrame >= 0 && currentFrame < arr.size()) {
        JSONObject obj = arr.getJSONObject(currentFrame);
        currentPSNR = (float) obj.getDouble("psnr");
        currentSSIM = (float) obj.getDouble("ssim");
      } else {
        currentPSNR = avgPSNR;
        currentSSIM = avgSSIM;
      }
    } else {
      currentPSNR = avgPSNR;
      currentSSIM = avgSSIM;
    }
  } catch (Exception e) {
    currentPSNR = avgPSNR;
    currentSSIM = avgSSIM;
  }
  
  // Draw overlay
  fill(0, 200);
  rect(10, 10, 320, 170, 5);
  
  fill(255);
  textSize(14);
  textAlign(LEFT, TOP);
  
  text("NERF METRICS", 20, 20);
  text("Frame: " + currentFrame + " / " + (totalFrames-1), 20, 45);
  text("PSNR: " + nf(currentPSNR, 0, 2) + " dB", 20, 70);
  text("SSIM: " + nf(currentSSIM, 0, 4), 20, 95);
  text("------------------------", 20, 120);
  text("AVG PSNR: " + nf(avgPSNR, 0, 2) + " dB", 20, 140);
  text("AVG SSIM: " + nf(avgSSIM, 0, 4), 20, 160);
}

// ================= INFO PANEL (UPDATED - TEXT LEBIH BESAR) =================
void displayInfoPanel() {
  fill(0, 200);  // Background lebih opaque
  rect(width - 350, 10, 340, 280, 5);  // Box lebih besar
  
  fill(255);
  textSize(16);  // Ukuran judul lebih besar
  textAlign(LEFT, TOP);
  
  // Judul
  text("CONTROLS:", width - 340, 20);
  
  textSize(13);  // Ukuran text controls
  text("• Mouse Drag: Rotate", width - 340, 45);
  text("• Mouse Wheel: Zoom", width - 340, 65);
  text("• Space: Play/Pause", width - 340, 85);
  text("• Arrow Keys: Frames", width - 340, 105);
  
  // Separator
  stroke(255, 100);
  strokeWeight(1);
  line(width - 340, 120, width - 20, 120);
  noStroke();
  
  // Info section
  textSize(16);
  text("INFO:", width - 340, 135);
  
  textSize(13);
  text("• Method: NeRF", width - 340, 160);
  text("• Training: 300 iter", width - 340, 180);
  text("• Resolution: 100x100", width - 340, 200);
  text("• Framework: PyTorch", width - 340, 220);
  
  // Separator
  stroke(255, 100);
  line(width - 340, 240, width - 20, 240);
  noStroke();
  
  // Author
  textSize(14);
  fill(255, 200);
  text("By: Jundulloh rizki ananda", width - 340, 255);
}

// ================= EVENT HANDLERS =================
void frameSlider(float val) {
  currentFrame = (int) val;
}

void playButton() {
  isPlaying = !isPlaying;
  Button btn = cp5.get(Button.class, "playButton");
  if (btn != null) btn.setLabel(isPlaying ? "Pause" : "Play");
}

void speedSlider(float val) { playbackSpeed = val; }
void zoomSlider(float val) { zoom = val; }

void mouseDragged() {
  if (mouseButton == LEFT) {
    rotationY += (mouseX - pmouseX) * 0.01;
    rotationX += (mouseY - pmouseY) * 0.01;
  }
}

void mouseWheel(MouseEvent event) {
  zoom += event.getCount() * 0.05;
  zoom = constrain(zoom, 0.1, 3.0);
  Slider s = cp5.get(Slider.class, "zoomSlider");
  if (s != null) s.setValue(zoom);
}

void keyPressed() {
  if (key == ' ') {
    isPlaying = !isPlaying;
    Button btn = cp5.get(Button.class, "playButton");
    if (btn != null) btn.setLabel(isPlaying ? "Pause" : "Play");
  } else if (keyCode == LEFT) {
    currentFrame = (currentFrame - 1 + totalFrames) % totalFrames;
    Slider s = cp5.get(Slider.class, "frameSlider");
    if (s != null) s.setValue(currentFrame);
  } else if (keyCode == RIGHT) {
    currentFrame = (currentFrame + 1) % totalFrames;
    Slider s = cp5.get(Slider.class, "frameSlider");
    if (s != null) s.setValue(currentFrame);
  }
}
