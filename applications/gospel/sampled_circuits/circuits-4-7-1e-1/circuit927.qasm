OPENQASM 2.0;
include "qelib1.inc";
qreg q928[4];
cx q928[2],q928[3];
rx(5*pi/4) q928[3];
cx q928[3],q928[2];
cx q928[2],q928[1];
cx q928[0],q928[1];
