OPENQASM 2.0;
include "qelib1.inc";
qreg q121[6];
cx q121[5],q121[4];
cx q121[4],q121[3];
cx q121[3],q121[2];
cx q121[2],q121[1];
cx q121[0],q121[1];
