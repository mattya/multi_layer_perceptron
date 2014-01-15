float beta = 1.0;
float lambda = 0.0000001;
float eta = -0.02;

int Wid = 20;
int mode = 0;

int NTrain = 0;
int NMax = 10000;

int Loop = 1000;

int N_layer = 2;
int[] N_neuron = {Wid*Wid, 30};
int NNMax = Wid*Wid+1;

float[][] d_in, d_out;

float[][] X;
float[][][] W;
float[][] B;
float[][] delta;

float sigmoid(float x){
  return 1.0/(1.0+exp(-beta*x));
}

float dsigmoid(float x){
  return beta*x*(1.0-x);
}

void alloc(){
  d_in = new float[NMax][N_neuron[0]];
  d_out = new float[NMax][N_neuron[N_layer-1]];
  
  X = new float[N_layer][NNMax];
  B = new float[N_layer][NNMax];
  delta = new float[N_layer][NNMax];
  W = new float[N_layer][NNMax][NNMax];
  println("alloc_done");
}

void set_zero(int n, float[] z){
  for(int i=0; i<n; i++) z[i] = 0;
}

void set_state(int n, float[] z, float[] x){
  for(int i=0; i<n; i++) z[i] = x[i];
}

void set_delta(int n, float[] z, float[] x, float[] d){
  for(int i=0; i<n; i++) d[i] = x[i]-z[i];
}

void random_init(){
  for(int i=0; i<N_layer; i++){
    if(i!=N_layer-1){
      for(int j=0; j<N_neuron[i+1]; j++){
        for(int k=0; k<N_neuron[i]; k++){
          W[i][j][k] = random(-0.1, 0.1);
        }
      }
    }
    for(int k=0; k<N_neuron[i]; k++) B[i][k] = random(-0.1, 0.1);
  }
}

void forward_prop(float[] l1, float[][] w, float[] l2, float[] b, int n1, int n2){
  for(int i=0; i<n2; i++){
    l2[i] = 0;
    float sum = 0;
    for(int j=0; j<n1; j++){
      sum += w[i][j]*l1[j];
    }
    sum += b[i];
    l2[i] = random(0, 1)<sigmoid(sum)?1:0;
  }
}

void back_prop(float[] l1, float[][] w, float[] l2, float[] b, int n1, int n2){
  for(int i=0; i<n1; i++){
    l1[i] = 0;
    float sum = 0;
    for(int j=0; j<n2; j++){
      sum += w[j][i]*l2[j];
    }
    sum += b[i];
    l1[i] = random(0, 1)<sigmoid(sum)?1:0;
  }
}

void update_weights(float[] l1, float[] l2, float[] l1_2, float[] l2_2, float[][] w, float[] b1, float[] b2, int n1, int n2){
  for(int i=0; i<n2; i++){
    for(int j=0; j<n1; j++){
      w[i][j] -= eta*(l2[i]*l1[j] - l2_2[i]*l1_2[j]);
    }
  }
  for(int i=0; i<n1; i++){
    b1[i] -= eta*(l1[i] - l1_2[i]);
  }
  for(int i=0; i<n2; i++){
    b2[i] -= eta*(l2[i] - l2_2[i]);
  }
}

void train_step(int ind){
  for(int i=0; i<N_layer; i++){
    set_zero(N_neuron[i], X[i]);
    set_zero(N_neuron[i], delta[i]);
  }
  
  float[] nX0, nX1;
  nX0 = new float[N_neuron[0]];
  nX1 = new float[N_neuron[1]];
  set_state(N_neuron[0], X[0], d_in[ind]);
  forward_prop(X[0], W[0], X[1], B[1], N_neuron[0], N_neuron[1]);
  back_prop(nX0, W[0], X[1], B[0], N_neuron[0], N_neuron[1]);
  forward_prop(nX0, W[0], nX1, B[1], N_neuron[0], N_neuron[1]);
  
  update_weights(X[0], X[1], nX0, nX1, W[0], B[0], B[1], N_neuron[0], N_neuron[1]);
  
  
}

void train(){
//  random_init();
  for(int i=0; i<Loop; i++){
//    println(i);
    for(int j=0; j<NTrain; j++){
      train_step(j);
    }
  }
}


void setup(){
  size(Wid*10, Wid*10);
  colorMode(HSB, 100);
  background(100);
  
  alloc();
  frameRate(30);
}

void draw(){
  if(mode==1){
    forward_prop(X[0], W[0], X[1], B[1], N_neuron[0], N_neuron[1]);
    back_prop(X[0], W[0], X[1], B[0], N_neuron[0], N_neuron[1]);
  }
  for(int iy=0; iy<Wid; iy++){
    for(int ix=0; ix<Wid; ix++){
      fill(200*X[0][iy*Wid+ix]);
      rect(ix*10, iy*10, 10, 10);
    }
  }
//  ellipse(map(X[0][0], 0, 1, 0, width), map(X[0][1], 0, 1, 0, height), 3, 3);
}

int f = 0;

void mouseDragged(){
  X[0][(int)map(mouseY, 0, height, 0, Wid)*Wid + (int)map(mouseX, 0, width, 0, Wid)] = 1;
}

void mouseReleased(){
}

void keyPressed(){
  if(key=='t'){
    random_init();
    train();
    mode = 1;
  }else if(key=='c'){
    for(int i=0; i<N_neuron[0]; i++) X[0][i] = 0;
    mode = 0;
  }else if(key=='a'){
    for(int i=0; i<N_neuron[0]; i++){
      d_in[NTrain][i] = X[0][i];
    }
    NTrain++;
    println(NTrain);
  }else if(key=='m'){
    mode = 1;
  }
    /*
  f += 1;
  if(key=='q'){
    random_init();
    train();
    disp_func();
  }else if(key=='a'){
    lambda/=2;
    println("lambda="+lambda);
    random_init();
    train();
    disp_func();
  }else if(key=='d'){
    lambda*=2;
    println("lambda="+lambda);
    random_init();
    train();
    disp_func();
  }else if(key=='e'){
    train();
    disp_func();
    
  }
  */
}
