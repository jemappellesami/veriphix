OPENQASM 2.0;
include "qelib1.inc";
qreg q381[7];
cx q381[4],q381[5];
cx q381[4],q381[3];
cx q381[3],q381[2];
cx q381[2],q381[1];
cx q381[0],q381[1];
