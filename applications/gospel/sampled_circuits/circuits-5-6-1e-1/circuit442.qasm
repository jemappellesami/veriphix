OPENQASM 2.0;
include "qelib1.inc";
qreg q443[5];
cx q443[3],q443[4];
cx q443[2],q443[3];
cx q443[1],q443[2];
cx q443[1],q443[0];
rx(pi/4) q443[1];
