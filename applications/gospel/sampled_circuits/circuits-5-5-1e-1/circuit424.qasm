OPENQASM 2.0;
include "qelib1.inc";
qreg q425[5];
cx q425[2],q425[3];
cx q425[3],q425[4];
cx q425[3],q425[2];
cx q425[2],q425[1];
cx q425[0],q425[1];
