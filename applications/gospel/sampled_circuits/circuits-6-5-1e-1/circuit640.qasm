OPENQASM 2.0;
include "qelib1.inc";
qreg q641[6];
cx q641[1],q641[0];
rx(3*pi/2) q641[5];
cx q641[5],q641[4];
cx q641[3],q641[4];
cx q641[2],q641[3];
