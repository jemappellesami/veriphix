OPENQASM 2.0;
include "qelib1.inc";
qreg q259[3];
cx q259[1],q259[0];
rz(5*pi/4) q259[2];
cx q259[2],q259[1];
cx q259[0],q259[1];
