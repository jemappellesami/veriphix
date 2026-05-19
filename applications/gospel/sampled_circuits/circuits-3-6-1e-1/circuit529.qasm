OPENQASM 2.0;
include "qelib1.inc";
qreg q530[3];
cx q530[0],q530[1];
rx(5*pi/4) q530[2];
cx q530[1],q530[2];
cx q530[0],q530[1];
rx(pi/4) q530[1];
