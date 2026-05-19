OPENQASM 2.0;
include "qelib1.inc";
qreg q235[4];
rx(3*pi/4) q235[3];
cx q235[2],q235[3];
cx q235[2],q235[1];
cx q235[1],q235[0];
rx(pi/4) q235[1];
