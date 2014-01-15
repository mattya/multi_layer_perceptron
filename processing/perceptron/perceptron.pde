float beta = 10.0;
float lambda = 0.0000001;
float eta0 = 0.02;

int Wid = 4;

int NTrain = 0;
int NMax = 10000;

int Loop = 1000;

//int N_layer = 2;
//int[] N_neuron = {2, 1};
//int N_layer = 3;
//int[] N_neuron = {2, 30, 1};
int N_layer = 5;
int[] N_neuron = {2, 10, 10, 10, 1};
int NNMax = 101;

float[][] d_in, d_out;

float[][] X;
float[][][] W;
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
  for(int i=0; i<N_layer-1; i++){
    for(int j=0; j<N_neuron[i+1]; j++){
      for(int k=0; k<N_neuron[i]+1; k++){
        W[i][j][k] = random(-1, 1);
      }
    }
  }
}

void forward_prop(float[] l1, float[][] w, float[] l2, int n1, int n2){
  l1[n1-1] = 1.0;
  for(int i=0; i<n2; i++){
    l2[i] = 0;
    float sum = 0;
    for(int j=0; j<n1; j++){
      sum += w[i][j]*l1[j];
    }
    l2[i] = sigmoid(sum);
  }
}

// d2->d1
void back_prop(float[] l1, float[] d1, float[][] w, float[] d2, int n1, int n2){
  for(int i=0; i<n1-1; i++){
    d1[i] = 0;
    float sum = 0;
    for(int j=0; j<n2; j++){
      sum += w[j][i]*d2[j];
    }
    d1[i] = sum * dsigmoid(l1[i]);
  }
}

void update_weights(float[] l1, float[][] w, float[] d2, float eta, int n1, int n2){
  l1[n1-1] = 1.0;
  for(int i=0; i<n2; i++){
    for(int j=0; j<n1; j++){
//      print(w[i][j]+" ");
      w[i][j] -= eta*d2[i]*l1[j] + lambda*w[i][j];
    }
//    println();
  }
//  println();
}

void train_step(int ind){
  for(int i=0; i<N_layer; i++){
    set_zero(N_neuron[i], X[i]);
    set_zero(N_neuron[i], delta[i]);
  }
  
  set_state(N_neuron[0], X[0], d_in[ind]);
  for(int i=0; i<N_layer-1; i++){
    forward_prop(X[i], W[i], X[i+1], N_neuron[i]+1, N_neuron[i+1]);
  }
  set_delta(N_neuron[N_layer-1], d_out[ind], X[N_layer-1], delta[N_layer-1]);
  for(int i=N_layer-2; i>=1; i--){
    back_prop(X[i], delta[i], W[i], delta[i+1], N_neuron[i]+1, N_neuron[i+1]);
  }
  for(int i=0; i<N_layer-1; i++){
    update_weights(X[i], W[i], delta[i+1], eta0, N_neuron[i]+1, N_neuron[i+1]);
  }
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

void disp_func(){
  background(100);
  for(int i=0; i<width; i+=Wid){
    for(int j=0; j<height; j+=Wid){
      for(int k=0; k<N_layer; k++){
        set_zero(N_neuron[k]+1, X[k]);
        set_zero(N_neuron[k]+1, delta[k]);
      }
      X[0][0] = map(i, 0, width, -1, 1);
      X[0][1] = map(j, 0, height, -1, 1);
      for(int k=0; k<N_layer-1; k++){
        forward_prop(X[k], W[k], X[k+1], N_neuron[k]+1, N_neuron[k+1]);
      }
      fill(map(X[N_layer-1][0], 0, 1, 0, 60), 100, 100);
      noStroke();
      rect(i, j, Wid, Wid);
    }
  }
  
  for(int i=0; i<NTrain; i++){
    if((int)d_out[i][0]%2==0) fill(0);
    else fill(100);
    stroke(0);
    ellipse(map(d_in[i][0], -1, 1, 0, width), map(d_in[i][1], -1, 1, 0, height), 15, 15);
  }
  
  textSize(30);
  fill(40, 30, 100);
  for(int i=0; i<N_layer; i++){
    text(N_neuron[i], 30, 33*(i+1));
  }
    
}

void setup(){
  size(800, 800);
  colorMode(HSB, 100);
  background(100);
  
  alloc();
}

void draw(){
  
}

int f = 0;
void mousePressed(){
  println("Pressed");
  d_in[NTrain][0] = map(mouseX, 0, width, -1, 1);
  d_in[NTrain][1] = map(mouseY, 0, height, -1, 1);
  d_out[NTrain][0] = f%2;
  NTrain++;
  //train();
  //disp_func();
  for(int i=0; i<NTrain; i++){
    if((int)d_out[i][0]%2==0) fill(0);
    else fill(100);
    stroke(0);
    ellipse(map(d_in[i][0], -1, 1, 0, width), map(d_in[i][1], -1, 1, 0, height), 15, 15);
  }
}

void keyPressed(){
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
}
