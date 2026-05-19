OPENQASM 2.0;
include "qelib1.inc";
qreg q259[5];
rx(pi) q259[4];
cx q259[4],q259[3];
cx q259[3],q259[2];
cx q259[1],q259[2];
cx q259[0],q259[1];
