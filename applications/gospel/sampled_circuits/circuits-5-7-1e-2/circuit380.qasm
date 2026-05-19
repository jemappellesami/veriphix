OPENQASM 2.0;
include "qelib1.inc";
qreg q381[5];
rz(pi/2) q381[4];
cx q381[4],q381[3];
cx q381[2],q381[3];
cx q381[1],q381[2];
cx q381[0],q381[1];
